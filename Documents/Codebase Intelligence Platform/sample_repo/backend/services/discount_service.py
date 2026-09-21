from typing import Optional, Dict

class DiscountService:
    """
    Service responsible for validating coupons and calculating discount amounts.
    Used by CheckoutController and OrderService.
    """
    
    COUPON_DATABASE: Dict[str, float] = {
        "WELCOME10": 0.10,
        "SUMMER20": 0.20,
        "VIP50": 0.50
    }

    def calculate_discount(self, user_id: str, subtotal: float, coupon_code: Optional[str] = None) -> float:
        """
        Calculates total discount for an order.
        Line 18: Evaluates coupon rules and subtotal thresholds.
        """
        if subtotal <= 0:
            return 0.0
            
        discount = 0.0
        
        if coupon_code and coupon_code.upper() in self.COUPON_DATABASE:
            discount_rate = self.COUPON_DATABASE[coupon_code.upper()]
            discount = subtotal * discount_rate
            
        # Loyalty tier discount check
        if subtotal >= 100.0 and not coupon_code:
            discount = max(discount, subtotal * 0.05)
            
        return round(discount, 2)

    def validate_coupon(self, coupon_code: str) -> bool:
        """Checks if a coupon code is active and valid."""
        return coupon_code.upper() in self.COUPON_DATABASE
