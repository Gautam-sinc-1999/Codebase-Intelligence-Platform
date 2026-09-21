from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from backend.services.payment_service import PaymentService
from backend.services.discount_service import DiscountService
from backend.models.order import OrderModel

router = APIRouter(prefix="/api/checkout", tags=["checkout"])

class CheckoutRequest(BaseModel):
    user_id: str
    items: List[dict]
    coupon_code: Optional[str] = None
    payment_method: str

class CheckoutResponse(BaseModel):
    order_id: str
    status: str
    subtotal: float
    discount_amount: float
    tax_amount: float
    total_amount: float

@router.post("", response_model=CheckoutResponse)
async def process_checkout(request: CheckoutRequest):
    """
    Main checkout entry point called by frontend checkoutApi.js
    Calculates subtotal, applies discounts, computes tax, and processes payment.
    """
    subtotal = sum(item.get("price", 0) * item.get("quantity", 1) for item in request.items)
    
    # Calculate discount using DiscountService
    discount_service = DiscountService()
    discount_amount = discount_service.calculate_discount(
        user_id=request.user_id,
        subtotal=subtotal,
        coupon_code=request.coupon_code
    )
    
    taxable_amount = max(0.0, subtotal - discount_amount)
    tax_amount = round(taxable_amount * 0.08, 2)
    total_amount = round(taxable_amount + tax_amount, 2)
    
    # Process payment via PaymentService
    payment_service = PaymentService()
    payment_result = await payment_service.process_payment(
        user_id=request.user_id,
        amount=total_amount,
        payment_method=request.payment_method
    )
    
    if not payment_result.get("success"):
        raise HTTPException(status_code=400, detail=payment_result.get("error", "Payment failed"))
        
    # Save order to DB
    order = OrderModel.create_order(
        user_id=request.user_id,
        items=request.items,
        total_amount=total_amount,
        transaction_id=payment_result.get("transaction_id")
    )
    
    return CheckoutResponse(
        order_id=order.get("id"),
        status="completed",
        subtotal=subtotal,
        discount_amount=discount_amount,
        tax_amount=tax_amount,
        total_amount=total_amount
    )
