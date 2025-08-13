imageUpload.addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (file) {
        // Display original image
        const reader = new FileReader();
        reader.onload = function(e) {
            originalImage.src = e.target.result;
        }
        reader.readAsDataURL(file);

        // Send image to backend
        const formData = new FormData();
        formData.append('image', file);

        try {
            resultDiv.innerHTML = 'Processing...';
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                resultDiv.innerHTML = `Error: ${data.error}`;
                return;
            }
            
            heatmapImage.src = data.heatmap;
            const probability = (data.probability * 100).toFixed(2);
            resultDiv.innerHTML = `
                Prediction: ${data.prediction}<br>
                Confidence: ${probability}%
            `;
        } catch (error) {
            console.error('Fetch error:', error);
            resultDiv.innerHTML = `Error: ${error.message}`;
        }
    }
});