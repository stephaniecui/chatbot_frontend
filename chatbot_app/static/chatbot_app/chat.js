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
        
async function sendMessage(isRegenerate = false, messageToRegenerate = null) {
    let userInput;
    if (isRegenerate) {
        userInput = messageToRegenerate;
    } else {
        userInput = document.getElementById('user-input').value;
        if (userInput.trim() === '') return;
        appendMessage('user', userInput);
        document.getElementById('user-input').value = '';
    }
    
    const botMessageElement = appendMessage('bot', 'Impy is thinking hard...');
    
    try {
        const response = await fetch('/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ 
                message: userInput, 
                is_regenerate: isRegenerate
            })
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
                botMessageElement.querySelector('.message-bubble').innerHTML = formatResponse(result);
                scrollToBottom();
            }
        }
        
         // Add regenerate button after the response is complete
        if (!isRegenerate) {
            addRegenerateButton(botMessageElement, userInput);
        }
    } catch (error) {
        console.error('Error:', error);
        botMessageElement.querySelector('.message-bubble').innerHTML = 'Sorry, an error occurred while processing your request.';
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

function addRegenerateButton(messageElement, originalMessage) {
    const regenerateButton = document.createElement('button');
    regenerateButton.textContent = 'Regenerate';
    regenerateButton.classList.add('regenerate-button');
    regenerateButton.addEventListener('click', () => {
        // Remove the old message and regenerate button
        messageElement.remove();
        // Call sendMessage with isRegenerate flag and original message
        sendMessage(true, originalMessage);
    });
    messageElement.appendChild(regenerateButton);
}

function scrollToBottom() {
    const chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
}

window.onload = async function() {
    const botMessageElement = appendMessage('bot', '');
    try {
        const response = await fetch('/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ message: '' }) // Initial empty message to trigger the prompt
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
                botMessageElement.querySelector('.message-bubble').innerHTML = formatResponse(result);
                scrollToBottom();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        botMessageElement.querySelector('.message-bubble').innerHTML = 'Sorry, an error occurred while processing your request.';
    }
    scrollToBottom();
}

// Ensure the send button event listener is properly set
document.getElementById('send-button').addEventListener('click', () => sendMessage(false, null));

// Handle "Enter" key press for better user experience
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        sendMessage();
        e.preventDefault(); // Prevent newline in the textarea
    }
});

// New event listener for opening links in new tabs
document.getElementById('chat-box').addEventListener('click', function(e) {
    if (e.target.tagName === 'A') {
        e.preventDefault();
        window.open(e.target.href, '_blank');
    }
});
