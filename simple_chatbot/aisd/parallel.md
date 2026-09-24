## Service overview

A travel concierge that answers trip and itinerary questions by gathering several
facts at once before responding.

## Surface & capabilities

- **Parallel fact-gathering** — it can call multiple tools in a single turn and then
  synthesize the results into one answer.
- **Tools:**
  - `get_weather(city: str)` — returns a weather forecast for a city.
  - `get_events(city: str)` — lists notable events in a city.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Weather and events lookups only.** No booking, reservations, payments, or
  transactions of any kind.
- **No general-purpose actions or document search.**
- **No streaming.**
