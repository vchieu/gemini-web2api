#!/usr/bin/env node
/**
 * Worker model-routing regression test.
 *
 * Guards the bug where every request landed on the account default model
 * (3.1 Pro) no matter which model was asked for: the upstream ignores the
 * f.req inner[79] (family) / inner[80] (variant) model fields unless the
 * request also carries the browser-minted X-Goog-Ext-525001261-Jspb ticket,
 * and the ticket wins over the body when both are present.
 *
 * The Python package covers the same ground in tests/test_modular_sync.py
 * (ModelRoutingTests / ModelTicketTests); this file is the worker's mirror of
 * those checks, since worker.js is not importable as a module -- it exports a
 * default handler and talks to Cloudflare globals, so it is loaded through
 * `vm` with those globals stubbed out.
 *
 * Run it with:
 *
 *     node tests/worker_routing_test.js
 *
 * It exits 0 when everything passes and 1 otherwise, so it can be dropped
 * into any CI job that has Node available. No network access is needed: the
 * stubbed `fetch` throws, and only the pure routing helpers are exercised.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { webcrypto } = require('crypto');

const WORKER_PATH = path.join(__dirname, '..', 'cloudflare', 'worker.js');

// Tickets in the shape the browser mints them: index 14/15 carry the
// (family, variant) the upstream routes on. The thinking ones are the fresh
// captures (Extended variants); the rest are older captures of the standard
// models, which have kept working.
const TICKETS = {
  flash: '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,16,4,5,6,8,16],null,null,1,null,null,1,1,"561701FD-A2E8-4275-98B0-636DFE1554F9",null,null,[[6,908199999],[1789884088,624000000]]]',
  'flash-thinking': '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4,5,6,8,16,4,5,6,8,16],null,null,1,null,null,1,2,"BC3C18C3-7AF2-4803-B167-3EF53F6B8C06",null,null,[[5,870199999],[1791563601,466000000]]]',
  lite: '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,4,5,6,8],null,null,1,null,null,6,1,"32C786FF-9AE2-49E5-A67C-0A35421F63A6",null,null,[[null,95100000],[1789898801,131000000]]]',
  'lite-thinking': '[1,null,null,null,"cf41b0e0dd7d53e5",null,null,0,[4,5,6,8,16,4,5,6,8,16],null,null,1,null,null,6,2,"BC3C18C3-7AF2-4803-B167-3EF53F6B8C06",null,null,[[1,377099999],[1791563520,100000000]]]',
};

// ---------------------------------------------------------------------------
// Load the worker in a sandbox: everything Cloudflare provides is stubbed,
// and the default export (which needs the fetch handler signature) is dropped
// so only the top-level helpers remain.
// ---------------------------------------------------------------------------

const src = fs.readFileSync(WORKER_PATH, 'utf8').replace(/export default \{[\s\S]*$/, '');

const logs = [];
const context = {
  setTimeout,
  clearTimeout,
  AbortController,
  TextEncoder,
  TextDecoder,
  URLSearchParams,
  crypto: webcrypto,
  fetch: () => { throw new Error('network is stubbed out in this test'); },
  console: {
    log: (...a) => logs.push(a.join(' ')),
    warn: (...a) => logs.push(a.join(' ')),
    error: (...a) => logs.push(a.join(' ')),
  },
};
vm.createContext(context);
vm.runInContext(
  src + '\n; this.__routing = { MODELS, TICKET_HEADER, buildPayload, buildHeaders, ' +
  'ticketFor, resolveModel, upstreamEcho, checkRoutingLine };',
  context
);
const api = context.__routing;

function decodePayload(body) {
  const outer = JSON.parse(new URLSearchParams(body).get('f.req'));
  return JSON.parse(outer[1]);
}

// One complete wrb.fr data line, the shape upstream uses to echo the model
// that actually served the request (label at [42], family at [58], variant
// at [59]). The worker splits by newline before looking at a line.
function echoLine(label, family, variant) {
  const meta = new Array(60).fill(null);
  meta[42] = label;
  meta[58] = family;
  meta[59] = variant;
  return JSON.stringify([['wrb.fr', null, JSON.stringify(meta)]]);
}

(async () => {
  let failures = 0;
  const check = (name, actual, expected) => {
    const ok = JSON.stringify(actual) === JSON.stringify(expected);
    if (!ok) failures++;
    console.info((ok ? 'PASS ' : 'FAIL ') + name +
      (ok ? '' : ' -- got ' + JSON.stringify(actual) + ', want ' + JSON.stringify(expected)));
  };
  const warned = () => logs.some(l => l.indexOf('路由不匹配') !== -1);
  const reset = () => { logs.length = 0; };
  const cfg = { logRequests: true, modelTickets: TICKETS };

  // ── resolveModel: family/variant/ticket travel together ──────────────────
  const lite = api.resolveModel('gemini-3.5-flash-thinking-lite', 'gemini-3.6-flash', cfg);
  check('thinking-lite resolves to family 5, variant 2',
        [lite.modelId, lite.thinkMode, lite.variant], [5, 1, 2]);
  check('thinking-lite carries the lite-thinking ticket', lite.ticket, TICKETS['lite-thinking']);

  const flash = api.resolveModel('gemini-3.6-flash', 'gemini-3.6-flash', cfg);
  check('flash resolves to variant 1 with the flash ticket',
        [flash.variant, flash.ticket], [1, TICKETS.flash]);

  const auto = api.resolveModel('gemini-auto', 'gemini-3.6-flash', cfg);
  check('auto has no ticket (the upstream picks the model)', [auto.variant, auto.ticket], [1, null]);

  const noTickets = api.resolveModel('gemini-3.5-flash-thinking-lite', 'gemini-3.6-flash',
                                     { modelTickets: {} });
  check('missing modelTickets degrades to a null ticket, not an error',
        [noTickets.variant, noTickets.ticket], [2, null]);

  const unknown = api.resolveModel('some-client-model', 'gemini-flash-lite', cfg);
  check('unknown model falls back to the default model\'s ticket',
        [unknown.modelName, unknown.variant, unknown.ticket],
        ['some-client-model', 1, TICKETS.lite]);

  // ── buildPayload: inner[79] family + inner[80] variant ───────────────────
  check('payload carries family and variant',
        (() => { const i = decodePayload(api.buildPayload('hi', 5, 1, {}, 2)); return [i[79], i[80]]; })(),
        [5, 2]);
  check('payload carries the flash pair',
        (() => { const i = decodePayload(api.buildPayload('hi', 1, 4, {}, 1)); return [i[79], i[80]]; })(),
        [1, 1]);
  check('payload leaves the variant null when the model has none',
        (() => { const i = decodePayload(api.buildPayload('hi', 3, 4, {}, undefined)); return [i[79], i[80]]; })(),
        [3, null]);

  // ── buildHeaders: the ticket rides in its own header ─────────────────────
  const withTicket = await api.buildHeaders({}, TICKETS['lite-thinking']);
  check('ticket header is sent', withTicket[api.TICKET_HEADER], TICKETS['lite-thinking']);
  const withoutTicket = await api.buildHeaders({});
  check('no ticket header without a ticket', api.TICKET_HEADER in withoutTicket, false);

  // ── upstream echo: the only way to notice a silent misroute ──────────────
  check('upstream echo is parsed', api.upstreamEcho(echoLine('3.5 Flash-Lite Extended', 6, 2)),
        { label: '3.5 Flash-Lite Extended', family: 6, variant: 2 });
  check('non-response text has no echo', api.upstreamEcho('garbage'), null);

  reset();
  api.checkRoutingLine(echoLine('3.5 Flash-Lite Extended', 6, 2), TICKETS['lite-thinking'], 5, 2, cfg);
  check('matching ticket route logs nothing', warned(), false);

  // The ticket still advertises Extended (6,2) but it has aged out, so the
  // upstream serves the standard variant instead -- the observed symptom of
  // an expired ticket: the request succeeds, the model silently changes.
  reset();
  api.checkRoutingLine(echoLine('3.5 Flash-Lite', 6, 1), TICKETS['lite-thinking'], 5, 2, cfg);
  check('expired ticket (variant downgraded) is reported', warned(), true);

  reset();
  api.checkRoutingLine(echoLine('3.1 Pro', 3, 1), null, 5, 2, cfg);
  check('account default answering without a ticket is reported', warned(), true);

  reset();
  api.checkRoutingLine('not a data line', null, 5, 2, cfg);
  check('a line without an echo is ignored', warned(), false);

  console.info(failures === 0 ? '\nAll worker routing checks passed.'
                              : '\n' + failures + ' check(s) failed.');
  process.exit(failures === 0 ? 0 : 1);
})();
