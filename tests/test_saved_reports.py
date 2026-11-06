"""Report pages must render saved outcomes without contacting providers."""
import os
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


class SavedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'SCOREEDGE_PREWARM': '0',
                                          'SCOREEDGE_DATA_DIR': self.temp.name, 'ODDS_API_KEY': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        import app
        import settle
        self.site, self.settle = app, settle
        self.schedule = pd.DataFrame([{'home_team': 'KC', 'away_team': 'BUF',
            'gameday': '2026-11-01', 'week': 8, 'home_score': 28, 'away_score': 21}])
        self.main = {'day': '2026-11-01', 'year': '2026', 'game': 'Buffalo Bills @ Kansas City Chiefs',
            'mkt': 'moneyline', 'side': 'Kansas City Chiefs', 'line': None, 'am': -110,
            'edge': 3.0, 'away_ab': 'BUF', 'home_ab': 'KC', 'ts': 1}
        self.prop = {**self.main, 'mkt': 'player_passing_yards', 'side': 'Patrick Mahomes',
                     'player': 'Patrick Mahomes', 'ou': 'over', 'line': 250.5}

    def test_report_pages_render_real_saved_outcomes_without_network_or_jobs(self):
        site = self.site
        weekly = {'2026:8': {'ts': 0, 'rows': {'patrick mahomes': [{'team': 'KC', 'pass_yd': 300}]}}}
        with patch.object(site.data, '_get_cached', return_value=self.schedule), \
             patch.object(site.journal, 'load_journal', return_value=[self.main, self.prop]), \
             patch.object(site.picks, '_read', return_value=[{**self.main, 'status': 'pending'}]), \
             patch.object(self.settle, '_mem', {}), \
             patch.object(self.settle, '_read_cache', return_value=weekly), \
             patch.object(site.odds, '_cached', return_value=None), \
             patch('requests.sessions.Session.request') as network, \
             patch.object(site, 'ensure_data') as init, \
             patch.object(site, '_learn_tick') as learn, \
             patch.object(site.track, 'sync') as sync, \
             patch.object(site.picks, 'settle_pick') as settle_pick, \
             patch.object(site, '_full_slate_board') as slate:
            rows = site.backtest._settled(cached_only=True)
            self.assertEqual(len(rows), 2)
            self.assertEqual([r['status'] for r in rows], ['won', 'won'])
            with TestClient(site.app) as client:
                for path, title in [('/calibration?day=2026-11-01', 'CALIBRATION'),
                                    ('/backtest', 'BACKTEST'),
                                    ('/staking?bankroll=250&fraction=0.1&cap=0.02', 'STAKING')]:
                    with self.subTest(path=path):
                        start = time.monotonic()
                        response = client.get(path)
                        self.assertEqual(response.status_code, 200)
                        self.assertLess(time.monotonic() - start, 1)
                        self.assertIn(title, response.text)
                        self.assertNotIn('Preparing this page', response.text)
                        self.assertNotIn('x-page-cache', response.headers)
            for forbidden in (network, init, learn, sync, settle_pick, slate):
                forbidden.assert_not_called()

    def test_missing_scores_and_week_stats_stay_pending_and_do_not_poison_refresh(self):
        site = self.site
        with patch.object(site.data, '_get_cached', return_value=None), \
             patch.object(site.data, '_CACHE', {}), \
             patch.object(site.journal, 'load_journal', return_value=[self.main, self.prop]), \
             patch.object(self.settle, '_mem', {}), \
             patch.object(self.settle, '_read_cache', return_value={}), \
             patch.object(self.settle, '_fetch_week', return_value={'new': []}) as fetch, \
             patch.object(self.settle, '_write_cache'), \
             patch.object(site.data, '_get_retry') as network:
            rows = site.journal.settled_records(cached_only=True)
            self.assertEqual([r['status'] for r in rows], ['pending', 'pending'])
            network.assert_not_called()
            fetch.assert_not_called()
            self.assertEqual(self.settle._mem, {})
            self.assertEqual(self.settle._week_rows(2026, 8), {'new': []})
            fetch.assert_called_once_with(2026, 8)

    def test_duplicate_prices_share_one_result_calculation(self):
        with patch.object(self.site.journal, 'load_journal', return_value=[self.main] * 400), \
             patch.object(self.site.journal, '_game_result', return_value={'status': 'won'}) as grade:
            self.assertEqual(len(self.site.journal.settled_records(cached_only=True)), 400)
            grade.assert_called_once()

    def test_weekly_file_is_parsed_once_and_reloaded_when_changed(self):
        path = Path(self.temp.name) / 'weeks.json'
        path.write_text(json.dumps({'2026:8': {'rows': {'player': [{'pass_yd': 300}]}}}), encoding='utf-8')
        with patch.object(self.settle, '_CACHE', str(path)), \
             patch.object(self.settle.json, 'load', wraps=json.load) as read:
            for _ in range(500):
                self.assertEqual(self.settle._read_cache()['2026:8']['rows']['player'][0]['pass_yd'], 300)
            self.assertEqual(read.call_count, 1)
            path.write_text(json.dumps({'2026:8': {'rows': {'player': [{'pass_yd': 45}]}}}), encoding='utf-8')
            self.assertEqual(self.settle._read_cache()['2026:8']['rows']['player'][0]['pass_yd'], 45)
            self.assertEqual(read.call_count, 2)


if __name__ == '__main__':
    unittest.main()
