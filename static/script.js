async function loadOptions() {
  const res = await fetch('/options');
  const data = await res.json();
  const locSel = document.getElementById('location');
  const metroSel = document.getElementById('metro');

  locSel.innerHTML = '<option value="">Select a location</option>';
  data.locations.forEach(l => {
    const o = document.createElement('option');
    o.value = l; o.textContent = l;
    locSel.appendChild(o);
  });

  metroSel.innerHTML = '<option value="">Select</option>';
  data.nearby_metro.forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    metroSel.appendChild(o);
  });
}

const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const priceDiv = document.getElementById("price");
const explanationDiv = document.getElementById("ai-explanation");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  const payload = {
    "Bedrooms": document.getElementById("bedrooms").value,
    "Bathrooms": document.getElementById("bathrooms").value,
    "Area": document.getElementById("area").value,
    "Year Built": document.getElementById("yearBuilt").value,
    "Location": document.getElementById("location").value,
    "Nearby Metro": document.getElementById("metro").value
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    const formattedPrice = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(data.prediction);
    priceDiv.textContent = `${formattedPrice} lakhs`;
    explanationDiv.textContent = data.explanation || "No AI explanation available.";
    result.classList.remove("hidden");
  } catch (err) {
    priceDiv.textContent = "Error: " + err.message;
    explanationDiv.textContent = "";
    result.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Price";
  }
});

// Load dropdowns on page load
window.addEventListener('DOMContentLoaded', loadOptions);
