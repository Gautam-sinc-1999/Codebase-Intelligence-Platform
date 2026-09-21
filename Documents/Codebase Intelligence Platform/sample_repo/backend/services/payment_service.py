import uuid
from typing import Dict, Any

class PaymentService:
    """
    Handles payment gateway communication and transaction processing.
    Communicates with external payment APIs (Stripe / PayPal mock).
    """

    async def process_payment(self, user_id: str, amount: float, payment_method: str) -> Dict[str, Any]:
        """
        Executes payment charge against external provider.
        """
        if amount <= 0:
            return {"success": False, "error": "Invalid amount"}

        if payment_method not in ["credit_card", "paypal", "apple_pay"]:
            return {"success": False, "error": f"Unsupported payment method: {payment_method}"}

        transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
        return {
            "success": True,
            "transaction_id": transaction_id,
            "amount": amount,
            "status": "SETTLED"
        }
