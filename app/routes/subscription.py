"""
Subscription routes for Jolter.

Supports:
  - Promo code redemption (server-side validation)
  - Subscription status check
  - Purchase stub (creates active subscription without real payment)

When Stripe is ready, the /purchase endpoint gets replaced with a
Stripe Checkout redirect + webhook handler. The Subscription model
already has plan, status, and expires_at ready for real billing data.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User, Subscription, PromoCode
from app.routes.auth import get_current_user_required

r = APIRouter(prefix="/api/subscription", tags=["subscription"])


# --- Request / Response schemas ---

class RedeemRequest(BaseModel):
    code: str


class PurchaseRequest(BaseModel):
    plan: str  # "monthly" or "yearly"


class SubscriptionStatusResponse(BaseModel):
    has_subscription: bool
    plan: Optional[str] = None
    status: Optional[str] = None
    expires_at: Optional[str] = None


class RedeemResponse(BaseModel):
    status: str
    message: str


class PurchaseResponse(BaseModel):
    status: str
    message: str


class StatusResponse(BaseModel):
    status: str


# --- Endpoints ---

@r.get("/status", response_model=SubscriptionStatusResponse)
def get_status(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Check if the current user has an active subscription."""
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == user.id, Subscription.status == "active")
        .order_by(Subscription.created_at.desc())
        .first()
    )
    if not sub:
        return SubscriptionStatusResponse(has_subscription=False)

    # Check if expired
    if sub.expires_at and sub.expires_at < datetime.now(timezone.utc):
        sub.status = "expired"
        db.commit()
        return SubscriptionStatusResponse(has_subscription=False)

    return SubscriptionStatusResponse(
        has_subscription=True,
        plan=sub.plan,
        status=sub.status,
        expires_at=sub.expires_at.isoformat() if sub.expires_at else None,
    )


@r.post("/redeem", response_model=RedeemResponse)
def redeem_code(
    req: RedeemRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """
    Validate and redeem a promo code.
    Creates an active subscription if the code is valid.
    """
    code_str = (req.code or "").strip().upper()
    if not code_str:
        raise HTTPException(status_code=400, detail="Code is required")

    # Look up the code
    promo = (
        db.query(PromoCode)
        .filter(PromoCode.code == code_str, PromoCode.is_active == True)
        .first()
    )
    if not promo:
        raise HTTPException(status_code=404, detail="That code didn\u2019t work. Try another?")

    # Check usage limits (0 = unlimited)
    if promo.max_uses > 0 and promo.uses_count >= promo.max_uses:
        raise HTTPException(status_code=410, detail="This code has been fully redeemed")

    # Check if user already has an active subscription
    existing = (
        db.query(Subscription)
        .filter(Subscription.user_id == user.id, Subscription.status == "active")
        .first()
    )
    if existing:
        return RedeemResponse(status="ok", message="You already have an active subscription")

    # Check if user already used this specific code
    already_used = (
        db.query(Subscription)
        .filter(
            Subscription.user_id == user.id,
            Subscription.code_used == code_str,
        )
        .first()
    )
    if already_used:
        raise HTTPException(status_code=409, detail="You have already used this code")

    # Create subscription
    sub = Subscription(
        user_id=user.id,
        plan="code",
        status="active",
        code_used=code_str,
        expires_at=datetime.now(timezone.utc) + timedelta(days=365),
    )
    db.add(sub)

    # Increment usage count
    promo.uses_count += 1
    db.commit()

    return RedeemResponse(status="ok", message="Code accepted!")


@r.post("/purchase", response_model=PurchaseResponse)
def purchase(
    req: PurchaseRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """
    Stub purchase endpoint.

    For now, this creates an active subscription immediately without
    real payment. When Stripe is integrated, this endpoint will instead
    create a Stripe Checkout session and redirect the user. The webhook
    handler will then create the Subscription record on successful payment.
    """
    if req.plan not in ("monthly", "yearly"):
        raise HTTPException(status_code=400, detail="Plan must be 'monthly' or 'yearly'")

    # Check if user already has an active subscription
    existing = (
        db.query(Subscription)
        .filter(Subscription.user_id == user.id, Subscription.status == "active")
        .first()
    )
    if existing:
        return PurchaseResponse(status="ok", message="You already have an active subscription")

    # Set expiry based on plan
    if req.plan == "yearly":
        expires = datetime.now(timezone.utc) + timedelta(days=365)
    else:
        expires = datetime.now(timezone.utc) + timedelta(days=30)

    sub = Subscription(
        user_id=user.id,
        plan=req.plan,
        status="active",
        expires_at=expires,
    )
    db.add(sub)
    db.commit()

    return PurchaseResponse(status="ok", message="Subscription activated")


@r.post("/cancel", response_model=StatusResponse)
def cancel(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Cancel the user's active subscription."""
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == user.id, Subscription.status == "active")
        .first()
    )
    if not sub:
        raise HTTPException(status_code=404, detail="No active subscription found")

    sub.status = "cancelled"
    db.commit()
    return StatusResponse(status="ok")
