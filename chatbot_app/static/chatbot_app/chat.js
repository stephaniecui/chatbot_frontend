function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function formatResponse(response) {
    return response.replace(/\n/g, '<br>');
}

async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    const chatBox = document.getElementById('chat-box');

    if (userInput.trim() === '') return;
    
    appendMessage('user', userInput);
    document.getElementById('user-input').value = '';

    const responseContainer = document.createElement('div');
    responseContainer.classList.add('bot-message');
    chatBox.appendChild(responseContainer);

    try {
        const response = await fetch('/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ message: userInput })
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
                responseContainer.innerHTML = formatResponse(result);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function appendMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender + '-message');
    messageElement.innerHTML = `<div class="message-bubble">${message}</div>`;
    const chatBox = document.getElementById('chat-box');
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Ensure the send button event listener is properly set
document.getElementById('send-button').addEventListener('click', sendMessage);

// Optionally handle "Enter" key press for better user experience
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
        e.preventDefault(); // Prevent newline in the textarea
    }
});
