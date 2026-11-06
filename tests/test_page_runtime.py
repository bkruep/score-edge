"""Offline regression checks for slow-provider isolation and saved reports."""
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from starlette.middleware.gzip import GZipMiddleware

import page_runtime


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Background work did not finish')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        # An explicit workspace parent also supports restricted Windows runners.
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)

    def test_slow_page_is_single_flight_and_other_pages_respond(self):
        app = FastAPI()
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        gate = threading.Event()
        started = threading.Event()
        calls = []
        self.addCleanup(gate.set)

        @app.get('/dfs')
        def dfs(request: Request, sport: str = 'nfl'):
            calls.append(sport)
            started.set()
            gate.wait(3)
            return HTMLResponse('<body>' + sport * 5000 + '</body>')

        @app.get('/healthz')
        def health():
            return {'ok': True}

        with patch.dict(os.environ, {'SCOREEDGE_DATA_DIR': self.temp.name}):
            cache = page_runtime.install(app, Jinja2Templates(directory='templates'), lambda r: {'request': r})
        with TestClient(app) as client:
            start = time.monotonic()
            first = client.get('/dfs')
            self.assertLess(time.monotonic() - start, .5)
            self.assertEqual(first.headers['x-page-cache'], 'warming')
            self.assertTrue(started.wait(1))
            for _ in range(4):
                self.assertEqual(client.get('/dfs').headers['x-page-cache'], 'warming')
            page_runtime.warm_pages(app, ['/dfs'])
            self.assertEqual(calls, ['nfl'])
            self.assertEqual(client.get('/healthz').json(), {'ok': True})
            gate.set()
            eventually(lambda: cache.stats()['pending_pages'] == 0)
            ready = client.get('/dfs')
            self.assertEqual(ready.headers['x-page-cache'], 'hit')
            self.assertEqual(ready.headers['content-encoding'], 'gzip')
            self.assertLess(int(ready.headers['content-length']), len(ready.content) // 10)
            client.get('/dfs?sport=nba')
            eventually(lambda: len(calls) == 2 and cache.stats()['pending_pages'] == 0)
            self.assertIn('nba', client.get('/dfs?sport=nba').text)

    def test_timeout_returns_without_waiting_for_worker(self):
        gate = threading.Event()
        calls = []
        self.addCleanup(gate.set)

        def slow():
            calls.append(1)
            gate.wait(3)
            return 'done'

        start = time.monotonic()
        self.assertIsNone(page_runtime.bounded(slow, .03))
        self.assertIsNone(page_runtime.bounded(slow, .03))
        self.assertLess(time.monotonic() - start, .3)
        self.assertEqual(calls, [1])
        gate.set()
        eventually(lambda: all(f.done() for f in page_runtime._bounded_calls.values()))

    def test_failed_refresh_keeps_labeled_last_good_page_and_survives_restart(self):
        cache = page_runtime.PageCache(self.temp.name, ttl=.01)
        loading = lambda message: HTMLResponse(message, headers={'X-Page-Cache': 'warming'})
        cache.response('/dfs', {}, lambda: HTMLResponse('<body>saved lineup</body>'), loading)
        eventually(lambda: cache.stats()['pending_pages'] == 0)
        time.sleep(.02)

        def failure():
            raise RuntimeError('provider unavailable')

        restarted = page_runtime.PageCache(self.temp.name, ttl=.01)
        stale = restarted.response('/dfs', {}, failure, loading)
        self.assertEqual(stale.headers['x-page-cache'], 'stale')
        self.assertIn(b'saved lineup', stale.body)
        self.assertIn(b'prices may be outdated', stale.body)
        eventually(lambda: restarted.stats()['pending_pages'] == 0)
        self.assertEqual(restarted.stats()['refresh_failures'], 1)
        restarted.response('/dfs', {}, failure, loading)
        self.assertEqual(restarted.stats()['refresh_failures'], 1)

    def test_eviction_leaves_room_for_new_query_and_only_prunes_page_files(self):
        cache = page_runtime.PageCache(self.temp.name, max_entries=2)
        protected = Path(self.temp.name) / 'learning.json'
        protected.write_text('keep me', encoding='utf-8')
        for i in range(4):
            cache.response('/dfs', {'i': i}, lambda: HTMLResponse('board'), lambda m: HTMLResponse(m))
            eventually(lambda: cache.stats()['pending_pages'] == 0)
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(len(list(Path(self.temp.name).glob('page-*.json'))), 2)
        self.assertEqual(protected.read_text(encoding='utf-8'), 'keep me')

    def test_queue_rejects_unbounded_work(self):
        gate = threading.Event()
        started = threading.Event()
        pool = page_runtime.WorkPool(workers=1, capacity=1)
        self.addCleanup(gate.set)
        def wait():
            started.set()
            gate.wait(3)
        first = pool.submit(wait)
        self.assertTrue(started.wait(1))
        second = pool.submit(lambda: None)
        with self.assertRaises(page_runtime.queue.Full):
            pool.submit(lambda: None)
        gate.set()
        first.result(1)
        second.result(1)

    def test_real_model_page_never_fetches_or_trains_on_get(self):
        with patch.dict(os.environ, {'SCOREEDGE_PREWARM': '0', 'SCOREEDGE_DATA_DIR': self.temp.name,
                                     'ODDS_API_KEY': '', 'DFS_DATA_SOURCE': ''}):
            import app as site
            with patch.object(site, 'ensure_data', side_effect=AssertionError('initialization on GET')) as init, \
                 patch.object(site, '_learn_tick', side_effect=AssertionError('settlement on GET')) as settle, \
                 patch.object(site.model, 'load_state', side_effect=AssertionError('training on GET')) as train, \
                 patch.object(site.model, 'load_hunt_state', side_effect=AssertionError('hunt training on GET')) as hunts, \
                 patch.object(site, '_full_slate_board', side_effect=AssertionError('live slate on GET')) as slate, \
                 patch.object(site.odds, 'fetch_edge_slate', side_effect=AssertionError('network on GET')) as fetch:
                with TestClient(site.app) as client:
                    start = time.monotonic()
                    response = client.get('/model')
                    self.assertEqual(response.status_code, 200)
                    self.assertLess(time.monotonic() - start, 1)
                    self.assertIn('latest saved reports', response.text)
                    self.assertEqual(client.get('/api/runtime-status').status_code, 200)
                    for forbidden in (init, settle, train, hunts, slate, fetch):
                        forbidden.assert_not_called()


if __name__ == '__main__':
    unittest.main()
