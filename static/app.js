function predict(){
    const promptInput = document.getElementById("query").value;

    if(promptInput.length <= 0){
        alert("Please Input Prompt")
        return
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            prompt: promptInput
        })
    }).then(response => response.blob())
      .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        const imageElement = document.createElement("img");
        imageElement.src = imageUrl;
        const container = document.getElementById("image-container");
        container.appendChild(imageElement);
      });
}

const sendButton = document.querySelector('.input-group-append .btn');
sendButton.addEventListener('click', predict);