const API_KEY = "lefty-dev-key";
const sessionId = crypto.randomUUID();
const userId = "web-" + sessionId.slice(0, 8);

const thread = document.getElementById("thread");
const form = document.getElementById("form");
const input = document.getElementById("message");
const send = document.getElementById("send");

function showEmpty() {
  thread.innerHTML = `
    <div class="empty">
      What are you looking for?
    </div>`;
}

function addMessage(role, text, meta) {
  const empty = thread.querySelector(".empty");
  if (empty) empty.remove();

  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta-line";
    m.textContent = meta;
    el.appendChild(m);
  }
  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
}

function setBusy(busy) {
  send.disabled = busy;
  input.disabled = busy;
}

function setError(msg) {
  addMessage("bot", msg || "Something went wrong. Try again.");
}

async function chat(message) {
  setBusy(true);
  addMessage("user", message);
  try {
    const res = await fetch("/api/v1/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        session_id: sessionId,
        user_id: userId,
        message,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const bits = [];
    if (data.product_query) bits.push(data.product_query);
    if (data.budget != null) bits.push(`budget $${data.budget}`);
    if (data.candidate_count) bits.push(`${data.candidate_count} matches`);
    if (data.pick) {
      bits.push(
        `pick ${data.pick.retailer} $${Number(data.pick.total).toFixed(2)} total`
      );
    }
    addMessage("bot", data.reply, bits.join(" · ") || data.intent);
  } catch (e) {
    setError(String(e.message || e));
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  input.style.height = "auto";
  chat(message);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 128) + "px";
});

showEmpty();
input.focus();
