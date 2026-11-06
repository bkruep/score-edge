"""Bounded background rendering for public, read-only HTML pages.

Provider latency is never a reason to hold an HTTP connection open. Last good
pages can survive restarts on SCOREEDGE_DATA_DIR; stale pages are labeled.
"""
from concurrent.futures import Future
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
import time

from fastapi.responses import HTMLResponse

LOG=logging.getLogger(__name__)


class WorkPool:
    """Daemon threads with a bounded queue, including during server shutdown."""
    def __init__(self,workers=2,capacity=8):
        self.queue=queue.Queue(maxsize=capacity)
        for number in range(workers):
            threading.Thread(target=self._work,daemon=True,name=f'page-work-{number}').start()

    def _work(self):
        while True:
            future,fn=self.queue.get()
            try:
                if future.set_running_or_notify_cancel():
                    try:future.set_result(fn())
                    except BaseException as exc:future.set_exception(exc)
            finally:self.queue.task_done()

    def submit(self,fn):
        future=Future()
        self.queue.put_nowait((future,fn))
        return future


_bounded_pool=WorkPool(workers=2,capacity=2)
_bounded_lock=threading.Lock()
_bounded_calls={}


def bounded(fn,timeout,*args,**kwargs):
    """Timeout does not wait for executor shutdown; repeated calls share work."""
    key=(fn,repr(args),repr(sorted(kwargs.items())))
    with _bounded_lock:
        for old,future in list(_bounded_calls.items()):
            if future.done():_bounded_calls.pop(old,None)
        future=_bounded_calls.get(key)
        if future is None:
            try:future=_bounded_pool.submit(lambda:fn(*args,**kwargs))
            except queue.Full:return None
            _bounded_calls[key]=future
    try:return future.result(timeout=max(0.,timeout))
    except Exception:return None


class PageCache:
    def __init__(self,root,ttl=120,max_stale=3600,max_entries=24,max_bytes=12*1024*1024,pool=None):
        self.root=Path(root)
        self.ttl=ttl;self.max_stale=max_stale;self.max_entries=max_entries;self.max_bytes=max_bytes
        self.pool=pool or WorkPool()
        self.lock=threading.RLock();self.entries={};self.failures=0

    def _load(self,key):
        path=self.root/('page-'+key+'.json')
        try:
            if path.stat().st_size>3*1024*1024:return None
            value=json.loads(path.read_text(encoding='utf-8'))
            age=time.time()-float(value['built_at'])
            if 0<=age<=self.max_stale:
                return {'body':value['body'].encode('utf-8'),'built_at':float(value['built_at'])}
        except (OSError,ValueError,KeyError,TypeError):pass
        return None

    def _save(self,key,value):
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            path=self.root/('page-'+key+'.json');temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps({'built_at':value['built_at'],'body':value['body'].decode('utf-8')}),encoding='utf-8')
            os.replace(temp,path)
            # Only this disposable page-cache namespace is pruned. Never touch
            # predictions, results, settings, or any cloud/backup files.
            files=sorted(self.root.glob('page-*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
            for old in files[self.max_entries:]:
                if old.resolve().parent==self.root.resolve():old.unlink(missing_ok=True)
        except (OSError,ValueError):LOG.warning('Page cache persistence unavailable')

    def _trim(self,reserve=False):
        memory=sum(len((e.get('value') or {}).get('body',b'')) for e in self.entries.values())
        for key,entry in sorted(list(self.entries.items()),key=lambda kv:kv[1]['used_at']):
            if len(self.entries)<=self.max_entries-int(reserve) and memory<=self.max_bytes:break
            if not entry['running']:
                memory-=len((entry.get('value') or {}).get('body',b''));self.entries.pop(key,None)

    @staticmethod
    def _notice(body,message):
        banner=('<div role="status" style="position:relative;z-index:70;background:#102435;color:#e2e8f0;'
                'padding:10px 18px;font:13px system-ui;border-bottom:1px solid #334155">'+message+'</div>').encode()
        match=re.search(br'<body\b[^>]*>',body,re.I)
        if match:return body[:match.end()]+banner+body[match.end():]
        return banner+body

    def response(self,path,params,builder,loading):
        now=time.time()
        identity=json.dumps([path,params],sort_keys=True,default=str)
        key=hashlib.sha256(identity.encode()).hexdigest()
        with self.lock:
            entry=self.entries.get(key)
            if entry is None:
                self._trim(reserve=True)
                if len(self.entries)>=self.max_entries:
                    return loading('Other pages are refreshing. This page will retry shortly.')
                entry={'value':self._load(key),'running':False,'retry_at':0.,'used_at':now}
                self.entries[key]=entry
            entry['used_at']=now
            value=entry['value']
            fresh=bool(value and now-value['built_at']<self.ttl)
            if not fresh and not entry['running'] and now>=entry['retry_at']:
                entry['running']=True
                def build():
                    start=time.monotonic()
                    try:
                        result=builder()
                        if result.status_code!=200 or 'text/html' not in result.headers.get('content-type',''):
                            raise ValueError('Not a successful HTML page')
                        body=bytes(result.body)
                        if len(body)>2*1024*1024:raise ValueError('Page exceeds cache size limit')
                        saved={'body':body,'built_at':time.time()}
                        self._save(key,saved)
                        with self.lock:
                            entry.update(value=saved,running=False,retry_at=0.)
                            self._trim()
                        LOG.info('Rendered page %s in %.2fs',path,time.monotonic()-start)
                    except Exception as exc:
                        with self.lock:
                            entry.update(running=False,retry_at=time.time()+30)
                            self.failures+=1
                        LOG.warning('Page refresh failed for %s (%s)',path,type(exc).__name__)
                try:self.pool.submit(build)
                except queue.Full:entry.update(running=False,retry_at=now+3)
            if value and now-value['built_at']<=self.max_stale:
                body=value['body']
                state='hit' if fresh else 'stale'
                if not fresh:
                    when=datetime.fromtimestamp(value['built_at'],timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
                    body=self._notice(body,f'Showing a saved page built at {when}. A refresh is in progress; displayed prices may be outdated.')
                return HTMLResponse(body,headers={'X-Page-Cache':state})
        if entry['retry_at']>now:
            return loading('The data refresh is delayed. Retrying automatically; other tabs remain available.')
        return loading('Loading this page’s data in the background. It will refresh automatically.')

    def stats(self):
        with self.lock:
            return {'cached_pages':sum(bool(e['value']) for e in self.entries.values()),
                    'pending_pages':sum(e['running'] for e in self.entries.values()),'refresh_failures':self.failures}


def warm_pages(app,paths=None):
    """Queue the same default variants visitors request, without loopback I/O."""
    from starlette.requests import Request
    wanted=set(paths or ('/','/nfl','/nba','/touchdown','/goal-hunt','/homerun-hunt','/dfs'))
    host=os.environ.get('RENDER_EXTERNAL_HOSTNAME','127.0.0.1')
    for route in app.routes:
        if getattr(route,'path',None) not in wanted:continue
        try:
            request=Request({'type':'http','method':'GET','path':route.path,'query_string':b'',
                'headers':[(b'host',host.encode())],'scheme':'https','server':(host,443),'app':app})
            params={'request':request}
            for name,p in inspect.signature(route.dependant.call).parameters.items():
                if name!='request' and p.default is not inspect.Parameter.empty:params[name]=p.default
            route.dependant.call(**params)
        except Exception as exc:LOG.warning('Could not queue warm page %s (%s)',route.path,type(exc).__name__)


def start_warmer(app):
    def loop():
        time.sleep(8)
        while True:
            warm_pages(app)
            time.sleep(240)
    threading.Thread(target=loop,name='public-page-warmer',daemon=True).start()


PATHS={'/','/bets','/props','/touchdown','/dfs','/goal-hunt','/homerun-hunt','/racing','/golf',
       '/nfl','/nba','/cfb','/cbb','/mlb','/nhl','/mls','/calibration','/backtest','/staking',
       '/rankings','/trades','/draft-assistant','/waivers','/parlay','/edge-deck','/line-lab','/value-lab','/prop-radar'}

# These reports now read saved data only and must remain available even when
# both provider-render workers are occupied by unrelated slow requests.
PATHS -= {'/calibration','/backtest','/staking'}


def install(app,templates,context):
    root=Path(os.environ.get('SCOREEDGE_DATA_DIR') or Path(__file__).parent)/'.data_cache'/'page_responses'
    cache=PageCache(root)
    for route in app.routes:
        if getattr(route,'path',None) not in PATHS or 'GET' not in getattr(route,'methods',set()):continue
        original=route.dependant.call
        def wrap(fn,path):
            @functools.wraps(fn)
            def endpoint(*args,**kwargs):
                request=kwargs.get('request')
                if request is None:return fn(*args,**kwargs)
                params={k:v for k,v in kwargs.items() if k!='request'}
                def loading(message):
                    response=templates.TemplateResponse(request,'loading.html',{
                        **context(request),'message':message,'active_page':'loading'})
                    response.headers['X-Page-Cache']='warming'
                    response.headers['Retry-After']='5'
                    return response
                return cache.response(path,params,lambda:fn(*args,**kwargs),loading)
            return endpoint
        route.dependant.call=wrap(original,route.path)
    app.state.page_cache=cache
    return cache
