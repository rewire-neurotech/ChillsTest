"""
Music selector for Jolter.

Currently only one track is available (a_thousand_hearts).
This module is kept as a thin wrapper so the pipeline doesn't
break and new tracks can be added later without refactoring.
"""


def select_track(
    q1_low_voice: str = "",
    q2_chills: str = "",
    q3_first_call: str = "",
    q4_unseen: str = "",
) -> str:
    """
    Return the track name to use for this session.

    With a single track in the registry the answer is always
    'a_thousand_hearts'.  The function signature accepts the four
    question answers so the call-site doesn't need to change when
    multi-track selection is re-introduced later.
    """
    selected = "a_thousand_hearts"
    print(f"[MusicSelector] -> {selected} (single-track mode)")
    return selected
