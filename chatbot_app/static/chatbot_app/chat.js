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
    let formattedResponse = response.replace(/\n/g, '<br>');
    
    formattedResponse = formattedResponse.replace(/<br><br>/g, '</p><p>');
    
    formattedResponse = '<p>' + formattedResponse + '</p>';
    
    return formattedResponse;
}

let conversationHistory = [];

async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;
    
    appendMessage('user', userInput);
    document.getElementById('user-input').value = '';
    const botMessageElement = appendMessage('bot', '');
    
    try {
        const response = await fetch('/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ 
                message: userInput,
                history: conversationHistory
            })
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.trim() === '') continue;
                    const data = JSON.parse(line);
                    
                    if (data.status === 'start') {
                        // Clear any previous content
                        botMessageElement.querySelector('.message-bubble').innerHTML = '';
                    } else if (data.chunk) {
                        fullResponse += data.chunk;
                        botMessageElement.querySelector('.message-bubble').innerHTML = formatResponse(fullResponse);
                    } else if (data.status === 'end') {
                        // Response is complete
                        conversationHistory.push({ role: 'user', content: userInput });
                        conversationHistory.push({ role: 'assistant', content: fullResponse });
                    }
                }
                scrollToBottom();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        botMessageElement.querySelector('.message-bubble').innerHTML += '<br>Sorry, an error occurred while processing your request.';
    }
    scrollToBottom();
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.innerHTML = `<div class="message-bubble">${message}</div>`;
    chatBox.appendChild(messageElement);
    scrollToBottom();
    return messageElement;
}

function scrollToBottom() {
    const chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Ensure the send button event listener is properly set
document.getElementById('send-button').addEventListener('click', sendMessage);

// Handle "Enter" key press for better user experience
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        sendMessage();
        e.preventDefault(); // Prevent newline in the textarea
    }
});
