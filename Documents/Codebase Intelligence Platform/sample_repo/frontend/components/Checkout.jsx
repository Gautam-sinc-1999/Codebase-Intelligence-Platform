import React, { useState } from 'react';
import { submitCheckout } from '../services/checkoutApi';

export function CheckoutComponent({ cartItems, userId }) {
  const [coupon, setCoupon] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCheckout = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        user_id: userId,
        items: cartItems,
        coupon_code: coupon,
        payment_method: 'credit_card'
      };
      const response = await submitCheckout(payload);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="checkout-container">
      <h2>Order Checkout</h2>
      <input 
        type="text" 
        value={coupon} 
        onChange={(e) => setCoupon(e.target.value)} 
        placeholder="Enter coupon code (e.g. WELCOME10)" 
      />
      <button onClick={handleCheckout} disabled={loading}>
        {loading ? 'Processing...' : 'Place Order'}
      </button>

      {result && (
        <div className="success">
          Order Completed! Order ID: {result.order_id}, Total: ${result.total_amount}
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}
