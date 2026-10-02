const imageInput =
    document.getElementById("imageInput");

const dropZone =
    document.getElementById("dropZone");

const previewContainer =
    document.getElementById("previewContainer");

const preview =
    document.getElementById("preview");

const removeBtn =
    document.getElementById("removeBtn");

const classifyBtn =
    document.getElementById("classifyBtn");

const result =
    document.getElementById("result");

const resultImage =
    document.getElementById("resultImage");

const loading =
    document.getElementById("loading");


// ============================================================
// FILE SELECTION
// ============================================================

imageInput.addEventListener(
    "change",
    function () {

        if (this.files.length > 0) {

            showPreview(this.files[0]);

        }

    }
);


// ============================================================
// DRAG & DROP
// ============================================================

dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.classList.add("dragover");

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.classList.remove("dragover");

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.classList.remove("dragover");

        const files = event.dataTransfer.files;

        if (files.length > 0) {

            imageInput.files = files;

            showPreview(files[0]);

        }

    }
);


// ============================================================
// SHOW PREVIEW
// ============================================================

function showPreview(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select an image file.");

        return;

    }

    const imageURL =
        URL.createObjectURL(file);

    preview.src = imageURL;

    resultImage.src = imageURL;

    dropZone.style.display = "none";

    previewContainer.style.display = "block";

}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeBtn.addEventListener(
    "click",
    resetApp
);


// ============================================================
// CLASSIFY
// ============================================================

async function classifyImage() {

    const file = imageInput.files[0];

    if (!file) {

        alert(
            "Please upload a waste image first."
        );

        return;

    }


    const formData = new FormData();

    formData.append(
        "image",
        file
    );


    // Show loading screen

    loading.style.display = "flex";


    classifyBtn.disabled = true;


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (data.error) {

            alert(data.error);

            return;

        }


        // ================================================
        // MAIN RESULT
        // ================================================

        document.getElementById(
            "prediction"
        ).textContent =
            data.prediction.toUpperCase();


        const confidence =
            data.confidence;


        document.getElementById(
            "confidenceValue"
        ).textContent =
            `${confidence}%`;


        document.getElementById(
            "confidenceText"
        ).textContent =
            `${confidence}%`;


        // Confidence bar

        document.getElementById(
            "confidenceBar"
        ).style.width =
            `${confidence}%`;


        // ================================================
        // TOP 3
        // ================================================

        const container =
            document.getElementById(
                "topPredictions"
            );


        container.innerHTML = "";


        data.top_predictions.forEach(
            item => {

                const row =
                    document.createElement(
                        "div"
                    );


                row.className =
                    "prediction-row";


                row.innerHTML = `

                    <div class="prediction-info">

                        <span>
                            ${item.class}
                        </span>

                        <span>
                            ${item.confidence}%
                        </span>

                    </div>

                    <div class="small-progress-bg">

                        <div
                            class="small-progress"
                            style="
                                width: ${item.confidence}%
                            "
                        ></div>

                    </div>

                `;


                container.appendChild(row);

            }
        );


        // Show result

        result.style.display = "grid";


        // Scroll to result

        setTimeout(
            () => {

                result.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            },
            150
        );


    }

    catch (error) {

        console.error(error);

        alert(
            "Something went wrong while analyzing the image."
        );

    }

    finally {

        loading.style.display = "none";

        classifyBtn.disabled = false;

    }

}


// ============================================================
// RESET
// ============================================================

function resetApp() {

    imageInput.value = "";

    preview.src = "";

    resultImage.src = "";

    previewContainer.style.display = "none";

    dropZone.style.display = "flex";

    result.style.display = "none";

    document.getElementById(
        "confidenceBar"
    ).style.width = "0%";

}