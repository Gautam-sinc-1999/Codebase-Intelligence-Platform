import pytest
from backend.services.discount_service import DiscountService

def test_welcome10_coupon():
    service = DiscountService()
    discount = service.calculate_discount(user_id="user_123", subtotal=100.0, coupon_code="WELCOME10")
    assert discount == 10.0

def test_summer20_coupon():
    service = DiscountService()
    discount = service.calculate_discount(user_id="user_123", subtotal=200.0, coupon_code="SUMMER20")
    assert discount == 40.0

def test_loyalty_tier_discount():
    service = DiscountService()
    discount = service.calculate_discount(user_id="user_123", subtotal=150.0)
    assert discount == 7.5
