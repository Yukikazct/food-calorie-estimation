// Local browser integration check. No third-party Node packages required.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import assert from 'node:assert/strict';

const url = process.env.FOOD_APP_URL || 'http://127.0.0.1:8780/';
const output = resolve(process.env.FOOD_BROWSER_OUTPUT || 'runs/setup/browser');
const profile = await mkdtemp(join(tmpdir(), 'food-browser-'));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${profile}`, '--window-size=1200,1000', 'about:blank'
], {stdio: 'ignore'});
let socket;
try {
  let port;
  for (let attempt = 0; attempt < 100; attempt++) {
    try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { await pause(100); }
  }
  assert(port, 'Chrome did not start');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  const waiting = new Map(); let nextId = 0;
  socket.onmessage = event => {
    const value = JSON.parse(event.data);
    if (waiting.has(value.id)) {
      const {resolve, reject} = waiting.get(value.id); waiting.delete(value.id);
      value.error ? reject(Error(JSON.stringify(value.error))) : resolve(value.result);
    }
  };
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId; waiting.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const value = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (value.exceptionDetails) throw Error(JSON.stringify(value.exceptionDetails));
    return value.result.value;
  };
  const until = async expression => {
    for (let attempt = 0; attempt < 100; attempt++) {
      const value = await evaluate(expression); if (value) return value; await pause(100);
    }
    throw Error(`Timed out: ${expression}`);
  };
  await call('Page.enable');
  await call('Emulation.setDeviceMetricsOverride', {width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false});
  await call('Page.navigate', {url});
  await until("document.querySelector('#food')?.options.length === 11");
  const initial = await evaluate("({status:document.querySelector('#status').textContent, disabled:document.querySelector('#recognize').disabled, labels:[...document.querySelector('#food').options].slice(1).map(x=>x.text)})");
  assert(initial.disabled, 'Recognize must be disabled before selecting an image');
  assert(initial.labels.every(label => !label.includes('待补充')));
  const estimate = async grams => {
    await evaluate(`document.querySelector('#food').value='pizza';document.querySelector('#grams').value=${JSON.stringify(String(grams))};document.querySelector('#estimate').click()`);
    return until("document.querySelector('#result').textContent || (document.querySelector('#message').classList.contains('error') && document.querySelector('#message').textContent)");
  };
  const first = await estimate(100);
  const sources = await evaluate("[...document.querySelectorAll('#sources a')].map(x=>({text:x.textContent,url:x.href}))");
  assert(sources.length > 0); assert(sources.every(x => x.url.startsWith('https://fdc.nal.usda.gov/')));
  const second = await estimate(200);
  const values = text => [...text.matchAll(/[0-9]+\.?[0-9]*/g)].map(match => Number(match[0]));
  assert.deepEqual(values(second), values(first).map(n => n * 2));
  await mkdir(output, {recursive: true});
  const screenshot = await call('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await writeFile(join(output, 'calorie_page.png'), Buffer.from(screenshot.data, 'base64'));
  const invalid = await estimate(0); assert(invalid.includes('有效克数'));
  const oversized = await estimate(6000); assert(oversized.includes('5000'));
  const evidence = {checked_utc: new Date().toISOString(), status: initial.status, classes: initial.labels,
    grams_100: first, grams_200: second, sources, invalid_mass: invalid, oversized_mass: oversized,
    scope: 'Real Chrome page controls and local API; does not establish classifier accuracy', passed: true};
  await writeFile(join(output, 'verification.json'), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence, null, 2));
} finally {
  if (socket) socket.close();
  chrome.kill('SIGTERM');
  await Promise.race([new Promise(resolve => chrome.once('exit', resolve)), pause(3000)]);
  if (chrome.exitCode === null && chrome.signalCode === null) chrome.kill('SIGKILL');
  await rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 200});
}
