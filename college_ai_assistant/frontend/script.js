/* =========================================================
   COLLEGE AI ASSISTANT
   Frontend Logic
   ========================================================= */

/* =========================================================
   DOM ELEMENTS
   ========================================================= */

const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const chatArea = document.getElementById("chatArea");

const newChat = document.getElementById("newChat");

const menuBtn = document.getElementById("menuBtn");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("overlay");

const welcome = document.getElementById("welcome");
const programmeBadge = document.getElementById("programmeBadge");

const programmeButtons = document.querySelectorAll(".programme-btn");

const quickLinks = document.querySelectorAll(".quick-link");

const questionCards = document.querySelectorAll(".question-card");

/* =========================================================
   STATE
   ========================================================= */

let selectedProgramme = "BCA";
let isSending = false;

/* =========================================================
   SIDEBAR
   ========================================================= */

function openSidebar() {
  sidebar.classList.add("open");
  overlay.classList.add("active");

  menuBtn.setAttribute("aria-expanded", "true");

  menuBtn.setAttribute("aria-label", "Close navigation");

  menuBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
}

function closeSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("active");

  menuBtn.setAttribute("aria-expanded", "false");

  menuBtn.setAttribute("aria-label", "Open navigation");

  menuBtn.innerHTML = '<i class="fa-solid fa-bars"></i>';
}

function toggleSidebar() {
  if (sidebar.classList.contains("open")) {
    closeSidebar();
  } else {
    openSidebar();
  }
}

menuBtn.addEventListener("click", toggleSidebar);

overlay.addEventListener("click", closeSidebar);

/* ESC closes sidebar */

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar.classList.contains("open")) {
    closeSidebar();
  }
});

/* =========================================================
   PROGRAMME SELECTION
   ========================================================= */

programmeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    programmeButtons.forEach((item) => {
      item.classList.remove("active");
    });

    button.classList.add("active");

    selectedProgramme = button.dataset.programme;

    programmeBadge.textContent = selectedProgramme;

    /*
     * Start a fresh conversation when
     * programme changes.
     */

    resetChat();

    /*
     * Close sidebar on mobile
     */

    if (window.innerWidth <= 850) {
      closeSidebar();
    }
  });
});

/* =========================================================
   QUICK ACCESS
   ========================================================= */

quickLinks.forEach((button) => {
  button.addEventListener("click", () => {
    const question = button.dataset.question;

    messageInput.value = question;

    messageInput.focus();

    autoResize();

    if (window.innerWidth <= 850) {
      closeSidebar();
    }
  });
});

/* =========================================================
   WELCOME QUESTION CARDS
   ========================================================= */

questionCards.forEach((button) => {
  button.addEventListener("click", () => {
    const question = button.dataset.question;

    messageInput.value = question;

    messageInput.focus();

    autoResize();
  });
});

/* =========================================================
   NEW CHAT
   ========================================================= */

newChat.addEventListener("click", resetChat);

function resetChat() {
  /*
   * Remove all messages.
   */

  chatArea.innerHTML = "";

  /*
   * Re-create welcome screen
   * instead of moving an old DOM node.
   */

  const welcomeClone = welcome.cloneNode(true);

  welcomeClone.style.display = "flex";

  welcomeClone.id = "welcome";

  chatArea.appendChild(welcomeClone);

  /*
   * Re-bind question cards
   */

  bindWelcomeQuestions(welcomeClone);

  /*
   * Clear input
   */

  messageInput.value = "";

  messageInput.style.height = "auto";

  /*
   * Close mobile sidebar
   */

  closeSidebar();

  messageInput.focus();
}

/* =========================================================
   WELCOME QUESTION BINDING
   ========================================================= */

function bindWelcomeQuestions(container) {
  const cards = container.querySelectorAll(".question-card");

  cards.forEach((button) => {
    button.addEventListener("click", () => {
      const question = button.dataset.question;

      messageInput.value = question;

      messageInput.focus();

      autoResize();
    });
  });
}

/* =========================================================
   SEND BUTTON
   ========================================================= */

sendBtn.addEventListener("click", sendMessage);

/* =========================================================
   ENTER KEY
   ========================================================= */

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();

    sendMessage();
  }
});

/* =========================================================
   TEXTAREA RESIZE
   ========================================================= */

messageInput.addEventListener("input", autoResize);

function autoResize() {
  messageInput.style.height = "auto";

  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

/* =========================================================
   SEND MESSAGE
   ========================================================= */

async function sendMessage() {
  const text = messageInput.value.trim();

  /*
   * Prevent empty messages.
   */

  if (!text) {
    return;
  }

  /*
   * Prevent multiple simultaneous requests.
   */

  if (isSending) {
    return;
  }

  isSending = true;

  /*
   * Hide welcome screen.
   */

  const currentWelcome = document.getElementById("welcome");

  if (currentWelcome) {
    currentWelcome.style.display = "none";
  }

  /*
   * Add user message.
   */

  addUserMessage(text);

  /*
   * Clear input.
   */

  messageInput.value = "";

  messageInput.style.height = "auto";

  /*
   * Disable send button.
   */

  sendBtn.disabled = true;

  /*
   * Show typing indicator.
   */

  const typingId = showTyping();

  try {
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        message: text,
        programme: selectedProgramme,
      }),
    });

    /*
     * Check server response.
     */

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();

    /*
     * Remove typing.
     */

    removeTyping(typingId);

    /*
     * Add AI response.
     */

    addAIMessage(data.response || "I could not generate a response.");
  } catch (error) {
    console.error("AI request failed:", error);

    removeTyping(typingId);

    addAIMessage(
      "Sorry, I could not connect to the AI server. Please make sure the FastAPI backend is running on port 8000.",
    );
  } finally {
    isSending = false;

    sendBtn.disabled = false;

    messageInput.focus();
  }
}

/* =========================================================
   USER MESSAGE
   ========================================================= */

function addUserMessage(text) {
  const message = document.createElement("div");

  message.className = "message user";

  const content = document.createElement("div");

  content.className = "message-content";

  /*
   * textContent prevents HTML injection.
   */

  content.textContent = text;

  message.appendChild(content);

  chatArea.appendChild(message);

  scrollToBottom();
}

/* =========================================================
   AI MESSAGE
   ========================================================= */

function addAIMessage(text) {
  const message = document.createElement("div");

  message.className = "message ai";

  const wrapper = document.createElement("div");

  wrapper.className = "ai-message";

  const avatar = document.createElement("div");

  avatar.className = "ai-avatar";

  avatar.textContent = "✦";

  const content = document.createElement("div");

  content.className = "message-content";

  /*
   * Keep AI response as plain text.
   */

  content.textContent = text;

  wrapper.appendChild(avatar);

  wrapper.appendChild(content);

  message.appendChild(wrapper);

  chatArea.appendChild(message);

  scrollToBottom();
}

/* =========================================================
   TYPING INDICATOR
   ========================================================= */

function showTyping() {
  const id = "typing-" + Date.now();

  const message = document.createElement("div");

  message.className = "message ai";

  message.id = id;

  message.innerHTML = `
        <div class="ai-message">

            <div class="ai-avatar">
                ✦
            </div>

            <div class="message-content">

                <div class="typing">

                    <span></span>
                    <span></span>
                    <span></span>

                </div>

            </div>

        </div>
    `;

  chatArea.appendChild(message);

  scrollToBottom();

  return id;
}

/* =========================================================
   REMOVE TYPING
   ========================================================= */

function removeTyping(id) {
  const element = document.getElementById(id);

  if (element) {
    element.remove();
  }
}

/* =========================================================
   SCROLL
   ========================================================= */

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTo({
      top: chatArea.scrollHeight,

      behavior: "smooth",
    });
  });
}

/* =========================================================
   WINDOW RESIZE
   ========================================================= */

window.addEventListener("resize", () => {
  /*
   * If screen becomes desktop,
   * remove mobile sidebar state.
   */

  if (window.innerWidth > 850 && sidebar.classList.contains("open")) {
    closeSidebar();
  }
});

/* =========================================================
   INITIALIZE
   ========================================================= */

function initialize() {
  /*
   * Make sure BCA is selected.
   */

  selectedProgramme = "BCA";

  programmeBadge.textContent = selectedProgramme;

  /*
   * Initial textarea height.
   */

  autoResize();
}

/* Start */

initialize();
