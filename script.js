const form = document.getElementById("issueForm");
const fileInput = document.getElementById("issuePhotos");
const previewGrid = document.getElementById("previewGrid");
const formMessage = document.getElementById("formMessage");
const submissionCard = document.getElementById("submissionCard");
const submissionSummary = document.getElementById("submissionSummary");
const submitButton = document.querySelector(".submit-button");
const backendStatus = document.getElementById("backendStatus");

const API_BASE = "http://127.0.0.1:8000";

let selectedFiles = [];
let imageURLs = [];
let currentImageIndex = 0;
let backendAvailable = false;
let ticketPollTimeout = null;

const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: "GET",
    });

    const result = await response.json();

    if (response.ok) {
      backendAvailable = true;
      const degraded = result?.status && result.status !== "ok";
      backendStatus.classList.toggle("offline", false);
      backendStatus.innerHTML = `
        <span class="status-dot"></span>
        <span>${degraded ? "Backend active - dependencies degraded" : "Backend active - ready for submission"}</span>
      `;
      submitButton.disabled = false;
    } else {
      throw new Error("Backend not responding");
    }
  } catch (error) {
    backendAvailable = false;
    backendStatus.classList.add("offline");
    backendStatus.innerHTML = `
      <span class="status-dot"></span>
      <span>Backend offline - please start the server</span>
    `;
    submitButton.disabled = true;
  }
};

const getSelectedValue = (name) => {
  const selected = document.querySelector(`input[name="${name}"]:checked`);
  return selected ? selected.value : "";
};

const removeFile = (indexToRemove) => {
  selectedFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
  imageURLs = imageURLs.filter((_, index) => index !== indexToRemove);
  renderPreviews(selectedFiles);
};

const createLargePreviewModal = () => {
  let modal = document.getElementById("imagePreviewModal");
  if (modal) return modal;

  modal = document.createElement("div");
  modal.id = "imagePreviewModal";
  modal.className = "image-preview-modal";
  modal.innerHTML = `
    <div class="modal-backdrop"></div>
    <div class="modal-content">
      <button class="close-modal" aria-label="Close preview">&times;</button>
      <div class="image-viewer">
        <img id="largeImage" src="" alt="Preview">
        <div class="image-counter"><span id="imageCounter">1/1</span></div>
      </div>
      <div class="image-navigation">
        <button class="nav-btn prev-btn" aria-label="Previous image">← Prev</button>
        <span id="imageName" class="image-name">Image Name</span>
        <button class="nav-btn next-btn" aria-label="Next image">Next →</button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);

  const closeBtn = modal.querySelector(".close-modal");
  const backdrop = modal.querySelector(".modal-backdrop");
  const prevBtn = modal.querySelector(".prev-btn");
  const nextBtn = modal.querySelector(".next-btn");

  const closeModal = () => {
    modal.style.display = "none";
    currentImageIndex = 0;
  };

  closeBtn.addEventListener("click", closeModal);
  backdrop.addEventListener("click", closeModal);

  prevBtn.addEventListener("click", () => {
    currentImageIndex = (currentImageIndex - 1 + imageURLs.length) % imageURLs.length;
    updateLargePreview();
  });

  nextBtn.addEventListener("click", () => {
    currentImageIndex = (currentImageIndex + 1) % imageURLs.length;
    updateLargePreview();
  });

  return modal;
};

const updateLargePreview = () => {
  const modal = document.getElementById("imagePreviewModal");
  if (!modal || imageURLs.length === 0) return;

  const largeImage = modal.querySelector("#largeImage");
  const counter = modal.querySelector("#imageCounter");
  const nameSpan = modal.querySelector("#imageName");

  largeImage.src = imageURLs[currentImageIndex].url;
  counter.textContent = `${currentImageIndex + 1}/${imageURLs.length}`;
  nameSpan.textContent = imageURLs[currentImageIndex].name;

  const prevBtn = modal.querySelector(".prev-btn");
  const nextBtn = modal.querySelector(".next-btn");
  prevBtn.disabled = imageURLs.length === 1;
  nextBtn.disabled = imageURLs.length === 1;
};

const showLargePreview = (index) => {
  currentImageIndex = index;
  const modal = document.getElementById("imagePreviewModal") || createLargePreviewModal();
  modal.style.display = "flex";
  updateLargePreview();
};

const renderPreviews = (files) => {
  previewGrid.innerHTML = "";
  imageURLs = [];

  if (!files.length) {
    return;
  }

  files.forEach((file, index) => {
    const imageURL = URL.createObjectURL(file);
    imageURLs.push({ url: imageURL, name: file.name });

    const tile = document.createElement("div");
    tile.className = "preview-tile";
    tile.style.cursor = "pointer";
    tile.style.position = "relative";

    const img = document.createElement("img");
    img.src = imageURL;
    img.alt = file.name;

    const label = document.createElement("span");
    label.className = "preview-name";
    label.textContent = file.name;

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-image-btn";
    removeBtn.setAttribute("aria-label", "Remove image");
    removeBtn.innerHTML = "&times;";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeFile(index);
    });

    tile.append(img, label, removeBtn);
    tile.addEventListener("click", () => showLargePreview(index));

    previewGrid.appendChild(tile);
  });
};

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");

const formatList = (items) => {
  if (!Array.isArray(items) || !items.length) {
    return "<p>Not provided</p>";
  }

  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
};

const formatBoolean = (value) => (value ? "Yes" : "No");
const formatJsonBlock = (value) =>
  value ? `<pre class="raw-output">${escapeHtml(JSON.stringify(value, null, 2))}</pre>` : "<p>Not available</p>";
const formatStatusHistory = (items) => {
  if (!Array.isArray(items) || !items.length) {
    return "<p>Not available</p>";
  }

  return `<ul>${items
    .map((item) => {
      const status = item?.status || "unknown";
      const step = item?.step || "unknown";
      const message = item?.message || "No message";
      const at = item?.at ? new Date(item.at).toLocaleString() : "Time unavailable";
      return `<li><strong>${escapeHtml(status)} / ${escapeHtml(step)}</strong><br>${escapeHtml(message)}<br><small>${escapeHtml(at)}</small></li>`;
    })
    .join("")}</ul>`;
};

const buildSummary = (data, ticket) => {
  const uploadedPhotoCount = selectedFiles.length;
  const apiReceivedImageCount = Number.isFinite(ticket?.receivedImageCount)
    ? ticket.receivedImageCount
    : 0;
  const backendImageCount = Number.isFinite(ticket?.analysis?.imageCount)
    ? ticket.analysis.imageCount
    : 0;
  const imagePipelineWarning =
    uploadedPhotoCount > 0 && (apiReceivedImageCount === 0 || backendImageCount === 0)
      ? "Warning: Images were selected in UI but were not fully processed by backend/model pipeline."
      : "";
  const structured = ticket?.analysis?.structured || null;
  const triage = ticket?.triage || {};
  const workflow = ticket?.workflow || {};
  const embedding = ticket?.embeddings?.summary || {};
  const rca = ticket?.rca || {};
  const flowDecision = rca?.result?.flowDecision || workflow?.rag?.decision || null;
  const rawOutput = ticket?.analysis?.rawOutput ? escapeHtml(ticket.analysis.rawOutput) : "No model output returned.";
  const latestStatus = ticket?.status || "processing";
  const latestStep = ticket?.currentStep || "received";
  const latestMessage = ticket?.statusMessage || "Waiting for processing update.";

  submissionSummary.innerHTML = `
    <strong>Processing Status</strong>
    <p>${escapeHtml(latestStatus)} · ${escapeHtml(latestStep)}</p>
    <p>${escapeHtml(latestMessage)}</p>
    <strong>Request ID</strong>
    <p>${escapeHtml(data.requestId)}</p>
    <strong>Submitter Email</strong>
    <p>${escapeHtml(data.userEmail)}</p>
    <strong>Request Type</strong>
    <p>${escapeHtml(data.requestType)}</p>
    <strong>Routing Path</strong>
    <p>${escapeHtml(data.primaryChoice)}</p>
    <strong>Submission Type</strong>
    <p>${escapeHtml(data.reviewType)}</p>
    <strong>Supporting Photos</strong>
    <p>${uploadedPhotoCount ? `${uploadedPhotoCount} file(s) selected in UI` : "No supporting photos selected"}</p>
    <strong>API Received Images</strong>
    <p>${apiReceivedImageCount} image(s) received by backend</p>
    <strong>Backend Image Processing</strong>
    <p>${backendImageCount} image(s) processed</p>
    ${imagePipelineWarning ? `<p class="warning-note">${escapeHtml(imagePipelineWarning)}</p>` : ""}
    <strong>Model</strong>
    <p>${escapeHtml(ticket?.analysis?.model || "Pending")}</p>
    <strong>Short Summary</strong>
    <p>${escapeHtml(triage?.summary || structured?.short_summary || "Pending analysis")}</p>
    <strong>Structured Problem</strong>
    <p>${escapeHtml(structured?.structured_problem || "Not provided")}</p>
    <strong>Error Type</strong>
    <p>${escapeHtml(triage?.errorType || structured?.error_type || "Not provided")}</p>
    <strong>System Context</strong>
    <p>${escapeHtml(triage?.systemContext || structured?.system_context || "Not provided")}</p>
    <strong>Page Context</strong>
    <p>${escapeHtml(triage?.pageContext || structured?.page_context || "Not provided")}</p>
    <strong>Error Code</strong>
    <p>${escapeHtml(triage?.errorCode || structured?.error_code || "Not provided")}</p>
    <strong>Severity</strong>
    <p>${escapeHtml(triage?.severity || structured?.severity || "Not provided")}${structured?.severity_weight ? ` (${escapeHtml(structured.severity_weight)})` : ""}</p>
    <strong>Impact Scope</strong>
    <p>${escapeHtml(triage?.impactScope || structured?.impact_scope || "Not provided")}</p>
    <strong>Related Issues</strong>
    ${formatList(triage?.relatedIssues || structured?.related_issues)}
    <strong>Image Evidence</strong>
    ${formatList(triage?.imageEvidence || structured?.image_evidence)}
    <strong>Impact Assessment</strong>
    <p>${escapeHtml(triage?.impactAssessment || structured?.impact_assessment || "Not provided")}</p>
    <strong>Preliminary Assessment</strong>
    <p>${escapeHtml(triage?.preliminaryAssessment || structured?.preliminary_assessment || "Not provided")}</p>
    <strong>Occurrence Hint</strong>
    <p>${escapeHtml(triage?.occurrenceHint || structured?.occurrence_hint || "Not provided")}</p>
    <strong>Data Gaps</strong>
    ${formatList(triage?.dataGaps || structured?.data_gaps)}
    <strong>Structuring Status</strong>
    <p>${escapeHtml(triage?.summary || structured?.short_summary ? "Completed" : "Pending")}</p>
    <strong>Embedding Status</strong>
    <p>${escapeHtml(embedding?.status || "Pending")}</p>
    <strong>Embedding Model</strong>
    <p>${escapeHtml(embedding?.model || ticket?.analysis?.embeddingModel || "Pending")}</p>
    <strong>Extraction Coverage</strong>
    <p>${escapeHtml(String(structured?.triage_signals?.extractedFieldCount ?? 0))} extracted signal(s)</p>
    <strong>RAG Routing</strong>
    <p>${escapeHtml(workflow?.rag?.status || "Not started")}</p>
    <strong>RAG Decision</strong>
    ${formatJsonBlock(flowDecision)}
    <strong>RAG Agent Trace</strong>
    ${formatJsonBlock(rca?.result?.agentMessages || workflow?.rag?.agentMessages || null)}
    <strong>Dedup Workflow</strong>
    <p>${escapeHtml(workflow?.dedup?.status || "Not started")}</p>
    <strong>Matched Incident</strong>
    <p>${escapeHtml(rca?.result?.matchedRequestId || workflow?.dedup?.matchedRecordId || "None")}</p>
    <strong>Duplicate Linked To Original Request</strong>
    <p>${escapeHtml(workflow?.dedup?.originalRequestId || rca?.result?.matchedRequestId || workflow?.dedup?.matchedRecordId || "None")}</p>
    <strong>Workflow Routing</strong>
    <p>${escapeHtml(workflow?.rca?.status || ticket?.rca?.status || "waiting_for_rag")}</p>
    <strong>RCA Source</strong>
    <p>${escapeHtml(rca?.result?.source || "Pending")}</p>
    <strong>OpenCode Exit Code</strong>
    <p>${escapeHtml(rca?.result?.exitCode ?? "Not available")}</p>
    <strong>OpenCode Timeout</strong>
    <p>${escapeHtml(rca?.result?.timedOut ? `Yes (${rca?.result?.timeoutSeconds ?? "unknown"}s)` : "No")}</p>
    <strong>OpenCode Terminated</strong>
    <p>${escapeHtml(rca?.result?.terminated ? "Yes" : "No")}</p>
    <strong>OpenCode Error</strong>
    <p>${escapeHtml(rca?.result?.error || "Not available")}</p>
    <strong>OpenCode Plan</strong>
    <pre class="raw-output">${escapeHtml(rca?.result?.fullPlan || rca?.result?.report || "Not available")}</pre>
    <strong>OpenCode STDERR</strong>
    <pre class="raw-output">${escapeHtml(rca?.result?.stderr || "Not available")}</pre>
    <strong>Status History</strong>
    ${formatStatusHistory(ticket?.statusHistory)}
    <strong>Raw Model Output</strong>
    <pre class="raw-output">${rawOutput}</pre>
  `;
};

const clearTicketPolling = () => {
  if (ticketPollTimeout) {
    clearTimeout(ticketPollTimeout);
    ticketPollTimeout = null;
  }
};

const scheduleTicketPoll = (requestId, payload, delayMs = 2500) => {
  clearTicketPolling();
  ticketPollTimeout = setTimeout(() => {
    pollTicketStatus(requestId, payload);
  }, delayMs);
};

const pollTicketStatus = async (requestId, payload) => {
  try {
    const response = await fetch(`${API_BASE}/api/tickets/${encodeURIComponent(requestId)}`, {
      method: "GET",
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Failed to fetch ticket status.");
    }

    const ticket = result.ticket;
    submissionCard.hidden = false;
    buildSummary(payload, ticket);

    if (ticket?.status === "completed") {
      formMessage.textContent = "Analysis completed and stored successfully.";
      formMessage.className = "form-message success";
      clearTicketPolling();
      return;
    }

    if (ticket?.status === "failed") {
      const failureMessage = ticket?.error?.message || ticket?.statusMessage || "Background processing failed.";
      formMessage.textContent = `Processing failed: ${failureMessage}`;
      formMessage.className = "form-message error";
      clearTicketPolling();
      return;
    }

    formMessage.textContent = ticket?.statusMessage || "Processing is still running...";
    formMessage.className = "form-message";
    scheduleTicketPoll(requestId, payload);
  } catch (error) {
    formMessage.textContent = `Unable to refresh processing status: ${error.message}`;
    formMessage.className = "form-message error";
    clearTicketPolling();
  }
};

const validateForm = (data) => {
  if (!data.requestId.trim()) {
    return "Request ID is required.";
  }

  if (!data.userEmail.trim()) {
    return "Submitter Email is required.";
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(data.userEmail.trim())) {
    return "Please enter a valid email address.";
  }

  if (!data.issueDescription.trim()) {
    return "Issue description is required.";
  }

  if (!data.primaryChoice) {
    return "Select either JDI or JGL.";
  }

  if (!data.reviewType) {
    return "Select PSUR, PADER, Literature Review, or Image Studio.";
  }

  return "";
};

fileInput.addEventListener("change", (e) => {
  if (e.target.files) {
    selectedFiles = selectedFiles.concat(Array.from(e.target.files));
    fileInput.value = "";
    renderPreviews(selectedFiles);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!backendAvailable) {
    formMessage.textContent = "Backend is not available. Please ensure the server is running.";
    formMessage.className = "form-message error";
    submissionCard.hidden = true;
    return;
  }

  const payload = {
    requestId: form.requestId.value,
    userEmail: form.userEmail.value,
    requestType: form.requestType.value,
    issueDescription: form.issueDescription.value,
    primaryChoice: getSelectedValue("primaryChoice"),
    reviewType: getSelectedValue("reviewType"),
  };

  const validationMessage = validateForm(payload);

  if (validationMessage) {
    formMessage.textContent = validationMessage;
    formMessage.className = "form-message error";
    submissionCard.hidden = true;
    return;
  }

  const formData = new FormData();
  formData.append("requestId", payload.requestId);
  formData.append("userEmail", payload.userEmail);
  formData.append("requestType", payload.requestType);
  formData.append("issueDescription", payload.issueDescription);
  formData.append("primaryChoice", payload.primaryChoice);
  formData.append("reviewType", payload.reviewType);

  selectedFiles.forEach((file) => {
    formData.append("issuePhotos", file);
  });

  submitButton.disabled = true;
  submitButton.textContent = "Submitting...";
  formMessage.textContent = "Submitting ticket to backend and running multimodal analysis...";
  formMessage.className = "form-message";

  try {
    const response = await fetch(`${API_BASE}/api/tickets/ingest`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Backend request failed.");
    }

    formMessage.textContent = result.message || "Request received. Processing started.";
    formMessage.className = "form-message success";
    submissionCard.hidden = false;
    buildSummary(payload, {
      status: result.status || "processing",
      currentStep: "received",
      statusMessage: result.message || "Request received. Processing started.",
      receivedImageCount: selectedFiles.length,
      analysis: {},
      triage: {},
      workflow: {},
      rca: {},
    });
    pollTicketStatus(result.requestId || payload.requestId, payload);
  } catch (error) {
    formMessage.textContent = `Submission failed: ${error.message}`;
    formMessage.className = "form-message error";
    submissionCard.hidden = true;
    clearTicketPolling();
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Submit and analyze";
  }
});

// Check backend health on page load
window.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  // Re-check health every 1 minute
  setInterval(checkBackendHealth, 60000);
});
