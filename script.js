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

const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: "GET",
    });

    if (response.ok) {
      backendAvailable = true;
      backendStatus.classList.remove("offline");
      backendStatus.innerHTML = `
        <span class="status-dot"></span>
        <span>Backend active - ready for submission</span>
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

const buildSummary = (data, backendResult) => {
  const uploadedPhotoCount = selectedFiles.length;
  const apiReceivedImageCount = Number.isFinite(backendResult?.receivedImageCount)
    ? backendResult.receivedImageCount
    : 0;
  const backendImageCount = Number.isFinite(backendResult?.imageCount)
    ? backendResult.imageCount
    : 0;
  const imagePipelineWarning =
    uploadedPhotoCount > 0 && (apiReceivedImageCount === 0 || backendImageCount === 0)
      ? "Warning: Images were selected in UI but were not fully processed by backend/model pipeline."
      : "";
  const structured = backendResult?.structured || null;
  const rawOutput = backendResult?.rawOutput ? escapeHtml(backendResult.rawOutput) : "No model output returned.";

  submissionSummary.innerHTML = `
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
    <p>${escapeHtml(backendResult?.model || "Unknown")}</p>
    <strong>Root Cause / Structured Problem</strong>
    <p>${escapeHtml(structured?.structured_problem || "Not provided")}</p>
    <strong>Related Issues</strong>
    ${formatList(structured?.related_issues)}
    <strong>Image Evidence</strong>
    ${formatList(structured?.image_evidence)}
    <strong>Impact Assessment</strong>
    <p>${escapeHtml(structured?.impact_assessment || "Not provided")}</p>
    <strong>Preliminary Assessment</strong>
    <p>${escapeHtml(structured?.preliminary_assessment || "Not provided")}</p>
    <strong>Raw Model Output</strong>
    <pre class="raw-output">${rawOutput}</pre>
  `;
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
    return "Select PSUR, PADER, or Literature Review.";
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
    submissionCard.hidden = true;
  } catch (error) {
    formMessage.textContent = `Submission failed: ${error.message}`;
    formMessage.className = "form-message error";
    submissionCard.hidden = true;
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
