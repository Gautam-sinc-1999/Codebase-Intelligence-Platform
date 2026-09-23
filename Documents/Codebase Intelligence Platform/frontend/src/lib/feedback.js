/**
 * Which answers a reader is allowed to rate.
 *
 * Pulled out of the component because it is a rule, not a rendering concern, and getting it wrong
 * is quiet: a thumb offered on a half-streamed answer records a verdict on text that was still
 * arriving, and one offered on a turn with no `seq` posts to an address that does not exist.
 */

/**
 * True when `message` is a finished assistant answer that has been written to the thread.
 *
 * Four things disqualify a message, and each for its own reason:
 *
 * - a question — the reader wrote it, there is nothing to rate;
 * - an error — not an answer, and it has no trace behind it;
 * - a still-streaming answer — rating text that is still arriving rates something the reader has
 *   not finished reading;
 * - an answer with no `seq` — the turn has not been persisted, so there is no message to attach
 *   a rating to. `seq` is `0` for the first message in a thread, so this has to be an explicit
 *   integer check rather than a truthiness test.
 */
export function canRate(message, conversationId) {
  if (!message || !conversationId) return false;
  if (message.role !== 'assistant') return false;
  if (message.error || message.streaming) return false;
  return Number.isInteger(message.seq);
}

/**
 * The rating a click produces, given what is already recorded.
 *
 * Clicking the pressed thumb again keeps it rather than clearing it. A rating is a deliberate
 * statement; silently retracting it on a stray second click loses information that cannot be
 * recovered, and the reader can always press the other thumb to change their mind.
 */
export function nextRating(current, clicked) {
  return clicked;
}

/**
 * Whether a comment box should be offered for this click.
 *
 * Only on a thumbs-down, and only when it is a change of verdict — "what was wrong" is the part
 * worth knowing, and re-asking someone who has already answered is noise.
 */
export function shouldAskWhy(current, clicked) {
  return clicked === 'down' && current !== 'down';
}
