# Voice Call Platforms for AI Voice Assistant — Comparison

## 1. Telegram

### Feasibility: ✅ HIGH
- **pytgcalls / tgcalls** (MarshalX) — mature C++ Python extension + high-level SDK
- Works as a **userbot** (not Bot API — bots can't do calls)
- Supports: private incoming/outgoing calls, group voice chats
- Audio: PCM 16-bit 48kHz, can send/receive raw bytes directly from Python
- Can use Pyrogram or Telethon as MTProto backend

### How It Works
- Someone calls the userbot's Telegram account → `UpdatePhoneCall` raw update fires
- Userbot can accept (`phone.AcceptCall`) or discard (`phone.DiscardCall`) the call
- Once connected, bidirectional audio stream — pipe to STT → LLM → TTS → back

### Security Model
- **User ID check**: On `UpdatePhoneCall` with `PhoneCallRequested`, extract `participant_id` (the caller). Compare against Matej's Telegram user ID (`6305957121`). If mismatch → `phone.DiscardCall` immediately.
- **Telegram privacy settings**: Set "Who can call me" → "My Contacts" or "Nobody" in the userbot's privacy settings, then add only Matej as a contact. This is a first-line defense at Telegram's level.
- **Code-level whitelist**: Hardcode `ALLOWED_USER_IDS = {6305957121}` — reject everything else before even answering.
- **Voice chat (group calls)**: Create a private group with only Matej + the userbot. Nobody else can join unless invited.

### Risk Assessment
- **Low risk**: Telegram user IDs are immutable and unforgeable (server-verified). Caller ID comes from Telegram servers, not the client.
- **Attack vector**: Someone gets added to Matej's contacts on the userbot account → mitigated by code-level whitelist.
- **Verdict**: Very secure. Double layer (privacy settings + code whitelist) makes unauthorized calls nearly impossible.

---

## 2. WhatsApp

### Feasibility: ⚠️ MEDIUM (improving)
- **WhatsApp Business Calling via Twilio** — GA as of July 15, 2025
- Twilio Programmable Voice + WhatsApp = VoIP calls inside WhatsApp threads
- Designed for business-to-customer, but can be used for personal AI assistant
- **Cost**: Twilio pricing per minute + WhatsApp Business API fees
- **No unofficial/userbot approach** — WhatsApp aggressively bans unofficial clients

### How It Works
- Register a WhatsApp Business number via Twilio
- Consumer (Matej) initiates a call from WhatsApp to the business number
- Twilio receives the call, triggers webhook → your server handles audio via Twilio Media Streams
- Bidirectional audio: WebSocket stream from Twilio → STT → LLM → TTS → back to Twilio

### Security Model
- **Phone number whitelist**: In your webhook handler, check `From` number against Matej's phone number. Reject all others with `<Reject/>` TwiML.
- **Twilio-level restrictions**: Can configure Twilio to only accept calls from specific numbers.
- **WhatsApp verification**: The caller's phone number is verified by WhatsApp (tied to SIM/device).
- **Additional PIN/passphrase**: Could require a spoken passphrase before the bot activates.

### Risk Assessment
- **Medium risk**: Phone numbers can theoretically be SIM-swapped, but this is a targeted attack requiring significant effort.
- **Cost overhead**: Twilio charges per minute — not ideal for always-on personal assistant.
- **Latency**: Extra hop through Twilio infrastructure.
- **Verdict**: Feasible but expensive and more complex. Good as secondary option.

---

## 3. Signal

### Feasibility: ❌ LOW
- **No voice call API exists**. Signal has no Bot API at all (by design).
- `signal-cli` and `signalbot` only support **text messages**, not voice/video.
- Signal's voice calls use their own E2EE VoIP protocol — no hooks for bots.
- Signal's philosophy is anti-bot, anti-automation for voice.
- The only "bots" are text-based via signal-cli REST API wrapper.

### What's Possible
- Text-only chatbot via signal-cli → completely possible
- Voice notes (send/receive audio files as messages) → possible but not real-time
- Real-time voice calls → **not possible** without reverse-engineering the protocol

### Security Model
- N/A — can't build it.

### Verdict
- **Not viable** for voice assistant. Could only do async voice notes (user sends voice message → bot processes → sends voice message back). Not a real conversation.

---

## 4. Discord

### Feasibility: ✅ HIGH
- **discord.py / discord.js** — first-class voice channel support
- Bot joins voice channel, can play and record audio
- `discord.VoiceClient` — connect, play FFmpeg audio, receive audio via `discord.sinks`
- Real-time bidirectional audio streaming is well-supported

### How It Works
- Create a private Discord server (only Matej + bot)
- Matej joins a voice channel, bot auto-joins or is summoned
- Bot receives audio via sink → STT → LLM → TTS → plays back via FFmpegPCMAudio
- Existing projects: AssemblyAI's Discord Voice Bot tutorial shows this exact pattern

### Security Model
- **Private server**: Create a server with NO invite link. Only Matej is a member. Bot is added via OAuth2 with specific guild restriction.
- **Role-based**: Even if someone somehow joins, bot checks `interaction.user.id` against allowed Discord user ID.
- **Voice channel permissions**: Set voice channel permissions so only Matej's role can connect.
- **Server verification level**: Set to highest — requires verified email + phone.
- **Code-level check**: Before processing any audio, verify `member.id == MATEJ_DISCORD_ID`.

### Risk Assessment
- **Very low risk**: Private server with no invite link + role restrictions + code-level user ID check = triple layer.
- **Discord user IDs are immutable** — server-verified, can't be spoofed.
- **Attack vector**: Someone compromises Matej's Discord account → but that's true for any platform.
- **Bonus**: Discord has excellent audio quality and low latency for voice.
- **Free**: No per-minute costs.

---

## 5. General Security Model Comparison

| Factor | Telegram | WhatsApp | Signal | Discord |
|--------|----------|----------|--------|---------|
| Voice call feasibility | ✅ Yes (userbot) | ⚠️ Yes (Twilio) | ❌ No | ✅ Yes (bot) |
| User ID verification | ✅ Immutable TG ID | ✅ Phone number | N/A | ✅ Immutable Discord ID |
| Platform-level call restriction | ✅ Privacy settings | ⚠️ Via Twilio config | N/A | ✅ Server/channel perms |
| Code-level whitelist | ✅ Easy | ✅ Easy | N/A | ✅ Easy |
| Caller identity spoofable? | ❌ No (server-verified) | ⚠️ SIM swap risk | N/A | ❌ No (server-verified) |
| Cost | Free | $$/minute | N/A | Free |
| Latency | Low | Medium (Twilio hop) | N/A | Low |
| Audio quality | Good (48kHz) | Good | N/A | Excellent (Opus) |
| Matej already uses it? | ✅ Yes (primary) | ? | ? | ? |
| Complexity | Medium | High | N/A | Medium |

### Multi-Layer Security Approach (for any platform)

1. **Platform-level**: Privacy settings / server permissions to block unknown callers
2. **Code-level whitelist**: Hardcoded user ID check — reject before answering
3. **Voice passphrase** (optional): First 3 seconds of call require spoken passphrase
4. **Rate limiting**: Max 1 concurrent call, cooldown between calls
5. **Logging**: Log all call attempts (caller ID, timestamp) for audit
6. **Kill switch**: Ability to disable voice assistant instantly via text command

### The Core Threat
> Someone else calls the bot and gives it commands to execute on the computer.

**Mitigation layers:**
- Layer 1: They can't even reach the bot (platform restrictions)
- Layer 2: Even if they reach it, the bot rejects the call (code whitelist)
- Layer 3: Even if they somehow connect, voice passphrase required
- Layer 4: Even if they pass all that, sensitive commands require text confirmation
- Layer 5: All actions are logged and auditable

---

## 🏆 Recommendation

### Primary: **Telegram** 
- Already the main communication channel with the bot
- pytgcalls is mature and well-documented
- Free, low latency, secure user ID system
- Natural fit — Matej already talks to the bot on Telegram
- Privacy settings + code whitelist = bulletproof

### Secondary: **Discord**
- Best audio quality and most mature voice bot ecosystem
- Completely free, excellent security via private server
- Good fallback if Telegram voice has issues
- Slightly more setup (separate server) but rock-solid

### Not recommended:
- **WhatsApp**: Expensive (Twilio), complex setup, SIM-swap risk
- **Signal**: Not possible for voice calls

### Implementation Priority
1. Telegram voice calls via pytgcalls (Pyrogram backend since already used)
2. If needed later, Discord as alternative voice channel
