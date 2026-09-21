import test from 'node:test';
import assert from 'node:assert/strict';

import { driftBanner } from '../src/lib/drift.js';

const behind = {
  state: 'behind',
  name: 'psf/requests@main',
  branch: 'main',
  indexed_sha: 'aaaaaaaaaaaaaaaa',
  remote_sha: 'bbbbbbbbbbbbbbbb',
  acknowledged: false
};

test('a current repository shows nothing', () => {
  assert.equal(driftBanner({ state: 'current' }), null);
});

test('an unreachable remote shows nothing', () => {
  // Drift is advisory. A network failure is not worth interrupting someone's reading with.
  assert.equal(driftBanner({ state: 'unknown', error: 'Could not resolve host' }), null);
});

test('a missing report shows nothing', () => {
  assert.equal(driftBanner(null), null);
  assert.equal(driftBanner(undefined), null);
});

test('a moved branch offers the choice', () => {
  const banner = driftBanner(behind);
  assert.equal(banner.actionable, true);
  assert.match(banner.headline, /psf\/requests@main has moved on/);
  assert.equal(banner.detail, 'indexed aaaaaaa · latest bbbbbbb · main');
});

test('a commit the user already declined stays quiet', () => {
  assert.equal(driftBanner({ ...behind, acknowledged: true }), null);
});

test('a later commit reopens the question', () => {
  // Acknowledging one version is not silence forever: the server reports `acknowledged` against
  // the current remote sha, so a new commit arrives unacknowledged.
  const newer = { ...behind, remote_sha: 'ccccccccccccccc', acknowledged: false };
  assert.notEqual(driftBanner(newer), null);
});

test('a thread left behind is informed but offered no sync', () => {
  const banner = driftBanner({
    state: 'thread_behind',
    detail: 'This repository has been re-indexed since this thread’s last answer.'
  });
  assert.equal(banner.actionable, false, 'offered a sync that would do nothing');
  assert.equal(banner.severity, 'info');
  assert.match(banner.headline, /re-indexed/);
});

test('a deleted branch is a warning with no sync offered', () => {
  const banner = driftBanner({
    state: 'branch_gone',
    detail: "Branch 'main' no longer exists on the remote."
  });
  assert.equal(banner.severity, 'warning');
  assert.equal(banner.actionable, false, 'offered to pull from a branch that is gone');
});

test('a state with no detail still produces a readable headline', () => {
  const banner = driftBanner({ state: 'thread_behind' });
  assert.ok(banner.headline.length > 0);
});
