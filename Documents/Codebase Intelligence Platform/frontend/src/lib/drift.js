/**
 * Whether a drift report is worth interrupting the reader with, and how to phrase it.
 *
 * Kept as a plain module rather than living inside the component so it can be tested directly.
 * Deciding *not* to show the banner is the part most worth getting right: a banner that appears
 * when nothing is actionable is nagging, and a user who has been nagged stops reading banners.
 */

/** The five states the server can report. */
export const DRIFT_STATES = ['current', 'behind', 'thread_behind', 'branch_gone', 'unknown'];

/**
 * Returns null when nothing should be shown, otherwise a description of the banner.
 *
 * Silent cases, each for its own reason:
 *   current       nothing has changed
 *   unknown       GitHub was unreachable; drift is advisory and a network failure is not the
 *                 reader's problem
 *   acknowledged  the user already declined this exact commit. A *later* commit reopens the
 *                 question, because acknowledging one version is not silence forever
 */
export function driftBanner(drift) {
  if (!drift) return null;
  if (drift.state === 'current' || drift.state === 'unknown') return null;
  if (drift.state === 'behind' && drift.acknowledged) return null;

  if (drift.state === 'behind') {
    return {
      severity: 'action',
      actionable: true,
      headline: `${drift.name} has moved on since this thread began.`,
      detail: [
        `indexed ${(drift.indexed_sha || '').slice(0, 7)}`,
        `latest ${(drift.remote_sha || '').slice(0, 7)}`,
        drift.branch
      ].filter(Boolean).join(' · ')
    };
  }

  // thread_behind and branch_gone are informational. Neither offers a sync: in the first case
  // there is nothing upstream to pull, and in the second the branch no longer exists.
  return {
    severity: drift.state === 'branch_gone' ? 'warning' : 'info',
    actionable: false,
    headline: drift.detail || 'This thread describes an earlier version of the code.',
    detail: ''
  };
}
