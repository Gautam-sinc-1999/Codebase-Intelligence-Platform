/**
 * Frontend API client service for checkout endpoints.
 * Interacts with POST /api/checkout
 */
export async function submitCheckout(checkoutPayload) {
  const response = await fetch('/api/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(checkoutPayload)
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Checkout failed');
  }

  return await response.json();
}
