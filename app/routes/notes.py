"""
Sticky Notes routes for Jolter.

Supports:
  - CRUD for user's sticky notes
  - Jolt trigger (kicks off generation from note text)
  - State transitions (idle -> progress -> ready -> watched)
"""

from datetime import datetime, timezone, date
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import cfg
from app.db import get_db
from app.models import StickyNote, User, Subscription
from app.routes.auth import get_current_user_required
from app.utils.encryption import encrypt_field, decrypt_field

r = APIRouter(prefix="/api/notes", tags=["notes"])


# --- Request / Response schemas ---

class CreateNoteRequest(BaseModel):
    text: str = ""
    place: Optional[str] = None


class UpdateNoteRequest(BaseModel):
    text: str


class UpdateStateRequest(BaseModel):
    state: str  # idle, progress, ready, watched


class NoteResponse(BaseModel):
    id: int
    text: str
    state: str
    place: Optional[str] = None
    jolt_count: int
    session_id: Optional[str] = None
    created_at: str
    updated_at: str


class NotesListResponse(BaseModel):
    notes: List[NoteResponse]
    total: int


class JoltResponse(BaseModel):
    status: str
    note_id: int
    can_jolt: bool
    reason: Optional[str] = None


class StatusResponse(BaseModel):
    status: str


# --- Helpers ---

def _note_to_response(note: StickyNote) -> NoteResponse:
    return NoteResponse(
        id=note.id,
        text=decrypt_field(note.text) or "",
        state=note.state,
        place=note.place,
        jolt_count=note.jolt_count,
        session_id=note.session_id,
        created_at=note.created_at.isoformat() if note.created_at else "",
        updated_at=note.updated_at.isoformat() if note.updated_at else "",
    )


def _check_jolt_permission(user: User, note: StickyNote, db: Session) -> tuple:
    """
    Check if the user can jolt this note.
    Returns (can_jolt: bool, reason: str or None).

    Free tier rules:
      - First jolt on a note is always free
      - Re-jolting a watched note requires an active subscription
    """
    # First jolt on this note is always free
    if note.jolt_count == 0:
        return True, None

    # Re-jolt requires subscription
    sub = (
        db.query(Subscription)
        .filter(Subscription.user_id == user.id, Subscription.status == "active")
        .first()
    )
    if sub:
        return True, None

    return False, "subscription_required"


def _reset_daily_jolt_count(user: User, db: Session):
    """Reset jolt_count_today if the date has changed since last jolt."""
    today = date.today()
    if user.last_jolt_date != today:
        user.jolt_count_today = 0
        user.last_jolt_date = today
        db.flush()


# --- Endpoints ---

@r.get("", response_model=NotesListResponse)
def list_notes(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """List all sticky notes for the current user, newest first."""
    notes = (
        db.query(StickyNote)
        .filter(StickyNote.user_id == user.id)
        .order_by(StickyNote.created_at.desc())
        .all()
    )
    return NotesListResponse(
        notes=[_note_to_response(n) for n in notes],
        total=len(notes),
    )


@r.post("", response_model=NoteResponse)
def create_note(
    req: CreateNoteRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Create a new sticky note."""
    note = StickyNote(
        user_id=user.id,
        text=encrypt_field(req.text) if req.text else "",
        state="idle",
        place=req.place,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return _note_to_response(note)


@r.put("/{note_id}", response_model=NoteResponse)
def update_note(
    note_id: int,
    req: UpdateNoteRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Update a sticky note's text."""
    note = (
        db.query(StickyNote)
        .filter(StickyNote.id == note_id, StickyNote.user_id == user.id)
        .first()
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.text = encrypt_field(req.text)
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)
    return _note_to_response(note)


@r.put("/{note_id}/state", response_model=NoteResponse)
def update_state(
    note_id: int,
    req: UpdateStateRequest,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Update a sticky note's state (idle, progress, ready, watched)."""
    valid_states = {"idle", "progress", "ready", "watched"}
    if req.state not in valid_states:
        raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of: {valid_states}")

    note = (
        db.query(StickyNote)
        .filter(StickyNote.id == note_id, StickyNote.user_id == user.id)
        .first()
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.state = req.state
    note.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(note)
    return _note_to_response(note)


@r.post("/{note_id}/jolt", response_model=JoltResponse)
def jolt_note(
    note_id: int,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """
    Trigger a jolt on a sticky note.

    Checks permissions (free tier vs subscription), then:
      - Sets note state to 'progress'
      - Increments jolt_count
      - Updates user's daily jolt tracking
      - Returns status so the frontend can start polling or proceed

    The actual generation (Claude + ElevenLabs + mix) will be triggered
    separately via the existing /api/demo/generate pipeline, with the
    note text as input.
    """
    note = (
        db.query(StickyNote)
        .filter(StickyNote.id == note_id, StickyNote.user_id == user.id)
        .first()
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if note.state == "progress":
        raise HTTPException(status_code=409, detail="Note is already being jolted")

    note_text = decrypt_field(note.text) or ""
    if not note_text.strip():
        raise HTTPException(status_code=400, detail="Cannot jolt an empty note")

    # Check permissions
    can_jolt, reason = _check_jolt_permission(user, note, db)
    if not can_jolt:
        return JoltResponse(
            status="blocked",
            note_id=note.id,
            can_jolt=False,
            reason=reason,
        )

    # Update note
    note.state = "progress"
    note.jolt_count += 1
    note.updated_at = datetime.now(timezone.utc)

    # Update user daily tracking
    _reset_daily_jolt_count(user, db)
    user.jolt_count_today += 1
    user.last_jolt_date = date.today()

    # Mark first jolt completed
    if not user.has_completed_first_jolt:
        user.has_completed_first_jolt = True

    db.commit()
    db.refresh(note)

    return JoltResponse(
        status="ok",
        note_id=note.id,
        can_jolt=True,
    )


@r.delete("/{note_id}", response_model=StatusResponse)
def delete_note(
    note_id: int,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db),
):
    """Delete a sticky note."""
    note = (
        db.query(StickyNote)
        .filter(StickyNote.id == note_id, StickyNote.user_id == user.id)
        .first()
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return StatusResponse(status="ok")
