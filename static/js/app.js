document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let socket;
    let currentTaskId = null;
    let taskHistory = [];
    
    // DOM elements
    const connectionStatus = document.getElementById('connectionStatus');
    const taskInput = document.getElementById('taskInput');
    const submitTaskBtn = document.getElementById('submitTaskBtn');
    const currentTaskSection = document.getElementById('currentTaskSection');
    const currentTaskDescription = document.getElementById('currentTaskDescription');
    const executionPlanList = document.getElementById('executionPlanList');
    const progressSteps = document.getElementById('progressSteps');
    const logContainer = document.getElementById('logContainer');
    const taskHistoryContainer = document.getElementById('taskHistoryContainer');
    
    // Connect to WebSocket
    function connectWebSocket() {
        // Generate a unique client ID
        const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
        
        // Connect to WebSocket server
        socket = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);
        
        socket.onopen = function(e) {
            console.log('WebSocket connection established');
            connectionStatus.innerHTML = `
                <span class="h-3 w-3 rounded-full bg-green-500 mr-2"></span>
                <span class="text-sm text-green-600">Connected</span>
            `;
            submitTaskBtn.disabled = false;
        };
        
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket connection closed');
            connectionStatus.innerHTML = `
                <span class="h-3 w-3 rounded-full bg-red-500 mr-2"></span>
                <span class="text-sm text-red-600">Disconnected</span>
            `;
            submitTaskBtn.disabled = true;
            
            // Attempt to reconnect after a delay
            setTimeout(connectWebSocket, 3000);
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket error:', error);
            connectionStatus.innerHTML = `
                <span class="h-3 w-3 rounded-full bg-red-500 mr-2"></span>
                <span class="text-sm text-red-600">Error</span>
            `;
        };
    }
    
    // Submit a new task
    async function submitTask() {
        const taskDescription = taskInput.value.trim();
        if (!taskDescription) {
            alert('Please enter a task description');
            return;
        }
        
        submitTaskBtn.disabled = true;
        submitTaskBtn.innerHTML = `
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
        `;
        
        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: taskDescription,
                    user_id: 'user_' + Math.random().toString(36).substr(2, 9)
                })
            });
            
            const data = await response.json();
            currentTaskId = data.task_id;
            
            // Clear current task display and prepare for new task
            currentTaskDescription.textContent = taskDescription;
            executionPlanList.innerHTML = '';
            progressSteps.innerHTML = '';
            currentTaskSection.classList.remove('hidden');
            
            // Add loading indicator until plan arrives
            executionPlanList.innerHTML = `
                <li class="pulse">Generating execution plan...</li>
            `;
            
            // Add log
            addLogEntry({
                agent: 'System',
                message: `Task submitted: ${taskDescription}`,
                log_type: 'info',
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error('Error submitting task:', error);
            alert('Failed to submit task. Please try again.');
            
            addLogEntry({
                agent: 'System',
                message: `Error submitting task: ${error.message}`,
                log_type: 'error',
                timestamp: new Date().toISOString()
            });
            
        } finally {
            submitTaskBtn.disabled = false;
            submitTaskBtn.innerHTML = 'Execute Task';
        }
    }
    
    // Handle WebSocket messages
    function handleWebSocketMessage(data) {
        console.log('Received message:', data);
        
        switch (data.type) {
            case 'log':
                addLogEntry(data);
                break;
                
            case 'plan':
                displayExecutionPlan(data.plan);
                break;
                
            case 'status_update':
                updateStepStatus(data);
                break;
                
            case 'task_complete':
                completeTask(data);
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    // Add a log entry
    function addLogEntry(log) {
        const logEntry = document.createElement('div');
        const timestamp = new Date(log.timestamp || Date.now()).toLocaleTimeString();
        
        let bgColor = 'bg-gray-100';
        let textColor = 'text-gray-800';
        let iconClass = 'fas fa-info-circle text-blue-500';
        
        switch (log.log_type) {
            case 'error':
                bgColor = 'bg-red-50';
                textColor = 'text-red-800';
                iconClass = 'fas fa-exclamation-circle text-red-500';
                break;
            case 'success':
                bgColor = 'bg-green-50';
                textColor = 'text-green-800';
                iconClass = 'fas fa-check-circle text-green-500';
                break;
            case 'warning':
                bgColor = 'bg-yellow-50';
                textColor = 'text-yellow-800';
                iconClass = 'fas fa-exclamation-triangle text-yellow-500';
                break;
        }
        
        logEntry.className = `${bgColor} ${textColor} p-2 rounded-md text-sm`;
        logEntry.innerHTML = `
            <div class="flex items-start">
                <div class="font-semibold mr-2">[${log.agent}]</div>
                <div class="flex-1">${log.message}</div>
                <div class="text-xs text-gray-500 ml-2">${timestamp}</div>
            </div>
        `;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    // Display execution plan
    function displayExecutionPlan(plan) {
     console.log("PLAN RECEIVED:", plan);
      executionPlanList.innerHTML = '';
        executionPlanList.classList.remove('pulse');
         const pulseElements = executionPlanList.querySelectorAll('.pulse');
pulseElements.forEach(el => el.remove());
console.log("executionPlanList element:", executionPlanList);
console.log("progressSteps element:", progressSteps);


     console.log("Starting displayExecutionPlan with plan:", plan);
        executionPlanList.innerHTML = '';
         if (plan && plan.length > 0) {
        plan.forEach((step, index) => {
        console.log(`Adding plan step ${index} to list`);
            const listItem = document.createElement('li');
            listItem.className = 'mb-1';
            listItem.textContent = step.description || step.action;
            executionPlanList.appendChild(listItem);
        });
        }
        console.log("Finished adding plan steps to list, now initializing progress steps");
        
        // Initialize progress steps
        progressSteps.innerHTML = '';
        plan.forEach((step, index) => {
        console.log(`Adding progress step ${index}`);
            const stepDiv = document.createElement('div');
            stepDiv.id = `step-${index}`;
            stepDiv.className = 'flex items-center';
            stepDiv.innerHTML = `
                <div class="flex items-center justify-center h-6 w-6 rounded-full bg-gray-200 text-xs text-gray-600 font-medium mr-2">
                    ${index + 1}
                </div>
                <div class="text-sm text-gray-600 flex-1">${step.description || step.action}</div>
                <div class="ml-2">
                    <span class="text-xs px-2 py-1 rounded-full bg-gray-200 text-gray-600">Pending</span>
                </div>
            `;
            progressSteps.appendChild(stepDiv);
        });
        console.log("Completed displayExecutionPlan function");
    }
    
    // Update step status
    function updateStepStatus(data) {
        const stepIndex = data.step_index;
        const status = data.status;
        
        if (stepIndex === undefined || !progressSteps.children[stepIndex]) {
            return;
        }
        
        const stepDiv = progressSteps.children[stepIndex];
        const statusSpan = stepDiv.querySelector('span');
        const stepIndicator = stepDiv.querySelector('div:first-child');
        
        if (status === 'in-progress') {
            statusSpan.className = 'text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-600';
            statusSpan.textContent = 'In Progress';
            stepIndicator.className = 'flex items-center justify-center h-6 w-6 rounded-full bg-blue-500 text-xs text-white font-medium mr-2';
            stepDiv.classList.add('pulse');
        } else if (status === 'completed') {
            statusSpan.className = 'text-xs px-2 py-1 rounded-full bg-green-100 text-green-600';
            statusSpan.textContent = 'Completed';
            stepIndicator.className = 'flex items-center justify-center h-6 w-6 rounded-full bg-green-500 text-xs text-white font-medium mr-2';
            stepDiv.classList.remove('pulse');
        } else if (status === 'failed') {
            statusSpan.className = 'text-xs px-2 py-1 rounded-full bg-red-100 text-red-600';
            statusSpan.textContent = 'Failed';
            stepIndicator.className = 'flex items-center justify-center h-6 w-6 rounded-full bg-red-500 text-xs text-white font-medium mr-2';
            stepDiv.classList.remove('pulse');
        }
    }
    
    // Complete task
    function completeTask(data) {
        // Add to task history
        addTaskToHistory({
            id: currentTaskId,
            description: currentTaskDescription.textContent,
            timestamp: new Date().toISOString()
        });
        
        // Reset current task
        setTimeout(() => {
            currentTaskId = null;
            taskInput.value = '';
        }, 5000);
    }
    
    // Add task to history
    function addTaskToHistory(task) {
        taskHistory.unshift(task);
        
        // Only keep last 10 tasks
        if (taskHistory.length > 10) {
            taskHistory.pop();
        }
        
        // Update display
        updateTaskHistory();
    }
    
    // Update task history display
    function updateTaskHistory() {
        taskHistoryContainer.innerHTML = '';
        
        if (taskHistory.length === 0) {
            taskHistoryContainer.innerHTML = '<div class="text-gray-500 text-center py-8">No tasks completed yet</div>';
            return;
        }
        
        taskHistory.forEach(task => {
            const date = new Date(task.timestamp).toLocaleString();
            const card = document.createElement('div');
            card.className = 'task-card bg-white p-4 rounded-lg shadow-sm border border-gray-200';
            card.innerHTML = `
                <p class="font-medium text-gray-800 mb-2 truncate">${task.description}</p>
                <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">${date}</span>
                    <button class="text-xs text-blue-600 hover:text-blue-800" data-task-id="${task.id}">View Details</button>
                </div>
            `;
            taskHistoryContainer.appendChild(card);
        });
    }
    
    // Event listeners
    submitTaskBtn.addEventListener('click', submitTask);
    
    taskInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            submitTask();
        }
    });
    
    // Initial setup
    connectWebSocket();
});