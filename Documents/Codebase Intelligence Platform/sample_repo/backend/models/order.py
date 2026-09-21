import uuid
from datetime import datetime
from typing import Dict, List, Any

class OrderModel:
    """
    Data Access Model for 'orders' database table.
    Interacts with SQL database table orders.
    """

    @staticmethod
    def create_order(user_id: str, items: List[Dict], total_amount: float, transaction_id: str) -> Dict[str, Any]:
        """
        Inserts new row into 'orders' table.
        """
        order_id = f"ord_{uuid.uuid4().hex[:8]}"
        return {
            "id": order_id,
            "user_id": user_id,
            "items": items,
            "total_amount": total_amount,
            "transaction_id": transaction_id,
            "created_at": datetime.utcnow().isoformat()
        }
