import test from 'node:test';
import assert from 'node:assert/strict';

import { canRate, nextRating, shouldAskWhy } from '../src/lib/feedback.js';

const answer = { role: 'assistant', content: 'It is in billing.py.', seq: 3 };

test('a finished answer can be rated', () => {
  assert.equal(canRate(answer, 'conv_1'), true);
});

test('the first message in a thread can be rated', () => {
  // seq 0 is falsy. A truthiness check here would make the opening answer of every thread
  // permanently unratable, which is exactly the kind of bug nobody reports.
  assert.equal(canRate({ role: 'assistant', seq: 0 }, 'conv_1'), true);
});

test('a question cannot be rated', () => {
  assert.equal(canRate({ role: 'user', content: 'where?', seq: 2 }, 'conv_1'), false);
});

test('an answer still streaming cannot be rated', () => {
  // Rating text that is still arriving rates something the reader has not finished reading.
  assert.equal(canRate({ ...answer, streaming: true }, 'conv_1'), false);
});

test('an error is not an answer', () => {
  assert.equal(canRate({ role: 'assistant', error: true, content: 'boom', seq: 3 }, 'conv_1'), false);
});

test('an unsaved turn cannot be rated', () => {
  // No seq means the turn never reached the thread, so there is no message to attach a rating to.
  assert.equal(canRate({ role: 'assistant', content: 'x' }, 'conv_1'), false);
  assert.equal(canRate({ ...answer, seq: undefined }, 'conv_1'), false);
});

test('nothing can be rated without a conversation', () => {
  assert.equal(canRate(answer, ''), false);
  assert.equal(canRate(answer, null), false);
});

test('a missing message is not ratable', () => {
  assert.equal(canRate(null, 'conv_1'), false);
});

test('clicking the pressed thumb keeps it rather than clearing it', () => {
  // A rating is a deliberate statement; a stray second click should not silently retract it.
  assert.equal(nextRating('up', 'up'), 'up');
  assert.equal(nextRating('down', 'down'), 'down');
});

test('clicking the other thumb changes the verdict', () => {
  assert.equal(nextRating('up', 'down'), 'down');
  assert.equal(nextRating('', 'up'), 'up');
});

test('the comment box is offered on a new thumbs-down only', () => {
  assert.equal(shouldAskWhy('', 'down'), true);
  assert.equal(shouldAskWhy('up', 'down'), true);
  // Already down — re-asking someone who has answered is noise.
  assert.equal(shouldAskWhy('down', 'down'), false);
  assert.equal(shouldAskWhy('', 'up'), false);
  assert.equal(shouldAskWhy('down', 'up'), false);
});
