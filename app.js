const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send");
const statusText = document.querySelector("#status");
const clearButton = document.querySelector("#clear");
const conversation = [];
let advisorState = {};
const ADVISOR_URL = "/advisor-v4";

function renderSafeText(container, text) {
  text = text.replace(/^#{1,6}\s+/gm, "");
  const parts = text.split(/(\[[^\]\n]+\]\\?\(https?:\/\/[^\s)]+\)|https?:\/\/[^\s<>]+|\*\*[^*]+\*\*)/g);
  parts.forEach((part) => {
    const markdownLink = part.match(/^\[([^\]\n]+)\]\\?\((https?:\/\/[^\s)]+)\)$/);
    const bareLink = part.match(/^https?:\/\/[^\s<>]+$/);
    if (markdownLink || bareLink) {
      const anchor = document.createElement("a");
      anchor.href = markdownLink ? markdownLink[2] : part;
      anchor.textContent = markdownLink ? markdownLink[1] : part;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      container.appendChild(anchor);
    } else if (part.startsWith("**") && part.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      container.appendChild(strong);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  });
}

function addMessage(role, text, tools = []) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "advisor-message"}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "A";
    avatar.setAttribute("aria-hidden", "true");
    article.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const paragraph = document.createElement("p");
  renderSafeText(paragraph, text);
  bubble.appendChild(paragraph);

  if (tools.length) {
    const verification = document.createElement("div");
    verification.className = "verification";
    verification.textContent = tools.length ? "Answered from BCIT records" : "";
    bubble.appendChild(verification);
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function payloadFor(question) {
  return {
    question,
    conversation: conversation.slice(-10),
    advisor_state: advisorState,
  };
}

async function askAdvisor(question) {
  addMessage("user", question);
  const priorConversation = conversation.slice();
  conversation.push({ role: "user", content: question });
  sendButton.disabled = true;
  statusText.textContent = "Asteris is checking academic rules…";
  const waiting = addMessage("assistant", "Checking verified Asteris data…");
  waiting.classList.add("typing");

  try {
    const payload = payloadFor(question);
    payload.conversation = priorConversation.slice(-10);
    const response = await fetch(ADVISOR_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    waiting.remove();
    if (!response.ok) throw new Error(data.detail || "The advisor could not answer.");
    addMessage("assistant", data.answer, data.tools_used || []);
    conversation.push({ role: "assistant", content: data.answer });
    if (data.advisor_state && typeof data.advisor_state === "object") {
      advisorState = data.advisor_state;
    }
    statusText.textContent = "Answer ready. Confirm important decisions with your institution.";
  } catch (error) {
    waiting.remove();
    const message = addMessage("assistant", error.message || "Asteris is temporarily unavailable.");
    message.classList.add("error");
    statusText.textContent = "The request was not completed.";
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sendButton.disabled) return;
  questionInput.value = "";
  askAdvisor(question);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.textContent;
    questionInput.focus();
  });
});

clearButton.addEventListener("click", () => {
  conversation.length = 0;
  advisorState = {};
  messages.querySelectorAll(".message:not(:first-child)").forEach((message) => message.remove());
  statusText.textContent = "New conversation started.";
  questionInput.focus();
});
