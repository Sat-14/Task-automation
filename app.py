"""
Multi-Agent Task Automation System
---------------------------------
A complete, self-contained application that allows users to describe tasks in natural language
and have them executed by a team of specialized agents.

Run with: python app.py
"""

import os
import json
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uvicorn
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
EMAIL_API_KEY = os.environ.get("EMAIL_API_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
BOOKING_API_KEY = os.environ.get("BOOKING_API_KEY", "")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. System will not function correctly!")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Task Automation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task storage
tasks_db: Dict[str, Dict[str, Any]] = {}
task_counter = 0

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TaskRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = "anonymous"

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections with clients."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """Disconnect a client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except RuntimeError:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

# Create a global instance of the connection manager
ws_manager = ConnectionManager()

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are TaskMaster, an advanced AI assistant that helps users accomplish tasks through a team of specialized agents.
Your primary responsibility is to understand the user's request, break it down into actionable steps, and assign each step to the appropriate agent."""

PLANNER_PROMPT_TEMPLATE = """
{system_prompt}

You need to create a detailed execution plan for the following request:
"{user_prompt}"

Available agents:
1. "KnowledgeAgent": Retrieves information from internal knowledge base (use for company-specific questions)
2. "SearchAgent": Performs web searches to find current information
3. "SlackAgent": Posts messages to specific Slack channels
4. "EmailAgent": Sends emails to specified recipients
5. "CommunicationAgent": Makes phone calls or sends text messages
6. "CalendarAgent": Checks, creates, updates or deletes calendar events
7. "WeatherAgent": Gets current weather or forecasts for specific locations
8. "BookingAgent": Books appointments, reservations, tickets, etc.

Create a JSON array of steps needed to complete this task. Each step should include:
- "agent": The name of the agent to use (from the list above)
- "action": A specific instruction for that agent
- "description": A user-friendly description of this step

Think step-by-step. If multiple agents are needed, arrange them in logical order.
If you need to store information from one step to use in another, indicate this in the action.

Example output format:
[
  {{
    "agent": "SearchAgent",
    "action": "Find top-rated Italian restaurants in downtown",
    "description": "Searching for Italian restaurants in the downtown area"
  }},
  {{
    "agent": "BookingAgent",
    "action": "Book a table for 2 at [restaurant_name] for tomorrow at 7pm",
    "description": "Making a reservation for 2 people"
  }},
  {{
    "agent": "CalendarAgent",
    "action": "Add dinner reservation to calendar for tomorrow at 7pm",
    "description": "Adding the reservation to your calendar"
  }}
]

Now create the execution plan:
"""

SLACK_PARSER_PROMPT_TEMPLATE = """
Extract the Slack channel and message content from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "channel": The Slack channel to post to (should start with #)
- "message": The message content to post

JSON Output:
"""

EMAIL_PARSER_PROMPT_TEMPLATE = """
Extract the email details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "to": The recipient's email address
- "subject": The email subject line
- "body": The email body content

JSON Output:
"""

EVENT_PARSER_PROMPT_TEMPLATE = """
Extract the calendar event details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "title": The event title
- "start_time": The start time in format "YYYY-MM-DD HH:MM"
- "end_time": The end time in format "YYYY-MM-DD HH:MM"
- "description": The event description
- "location": The event location (optional)
- "attendees": A list of email addresses for attendees (optional)

JSON Output:
"""

COMMUNICATION_PARSER_PROMPT_TEMPLATE = """
Extract the communication details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "method": Either "call" or "text"
- "recipient": The phone number or contact name
- "message": The message content (for texts) or talking points (for calls)

JSON Output:
"""

SEARCH_QUERY_PARSER_PROMPT_TEMPLATE = """
Extract a clear search query from the following action text:
"{action_text}"

Return just the search query text, nothing else.
"""

WEATHER_PARSER_PROMPT_TEMPLATE = """
Extract the weather request details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "location": The location to get weather for
- "type": Either "current" or "forecast"
- "days": Number of days for forecast (if applicable)

JSON Output:
"""

BOOKING_PARSER_PROMPT_TEMPLATE = """
Extract the booking details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "service_type": Type of booking (restaurant, hotel, flight, etc.)
- "provider": Name of provider if specified
- "details": All relevant details (date, time, number of people, etc.)
- "preferences": Any user preferences mentioned

JSON Output:
"""

# =============================================================================
# AGENT CLASSES
# =============================================================================

class BaseAgent:
    """Base class for all agents with common functionality."""
    
    def __init__(self, name):
        self.name = name
        logger.info(f"Initialized {name}")
    
    async def run(self, *args, **kwargs):
        """Execute the agent's main functionality."""
        raise NotImplementedError("Subclasses must implement run()")
    
    async def validate_credentials(self):
        """Check if all required credentials are available."""
        return True

class KnowledgeAgent(BaseAgent):
    """Agent for retrieving information from internal knowledge base."""
    
    def __init__(self, directory="knowledge_base"):
        super().__init__("KnowledgeAgent")
        self.directory = directory
        self.knowledge = self._load_knowledge()
    
    def _load_knowledge(self):
        """Load all knowledge from text files in the knowledge base directory."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            # Create a sample knowledge file
            with open(os.path.join(self.directory, "sample.txt"), "w") as f:
                f.write("This is sample knowledge. Replace with actual information.")
            
        knowledge_data = {}
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    knowledge_data[filename.replace(".txt", "")] = f.read()
                    
        return knowledge_data
    
    async def run(self, query: str) -> str:
        """Search the knowledge base for information related to the query."""
        logger.info(f"Searching knowledge base for: {query}")
        
        # In a real implementation, this would use a vector database or other search mechanism
        # For now, we'll implement a simple keyword-based search
        results = []
        query_terms = query.lower().split()
        
        for topic, content in self.knowledge.items():
            relevance_score = 0
            for term in query_terms:
                if term in content.lower():
                    relevance_score += content.lower().count(term)
            
            if relevance_score > 0:
                results.append((topic, content, relevance_score))
        
        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Return the most relevant information
        return f"Found information on '{results[0][0]}': {results[0][1][:500]}..."

class SearchAgent(BaseAgent):
    """Agent for performing web searches."""
    
    def __init__(self):
        super().__init__("SearchAgent")
    
    async def run(self, query: str) -> str:
        """Perform a web search for the given query."""
        logger.info(f"Searching the web for: {query}")
        
        try:
            # In a real implementation, this would use a search API
            # For demonstration purposes, we'll simulate search results
            await asyncio.sleep(1)
            
            # Simulated search results
            results = [
                {
                    "title": f"Result 1 for {query}",
                    "snippet": f"This is a sample search result for {query}.",
                    "link": "https://example.com/result1"
                },
                {
                    "title": f"Result 2 for {query}",
                    "snippet": f"Another sample result relevant to {query}.",
                    "link": "https://example.com/result2"
                },
                {
                    "title": f"Result 3 for {query}",
                    "snippet": f"A third sample result for the query {query}.",
                    "link": "https://example.com/result3"
                }
            ]
            
            # Format search results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(f"{i}. {result['title']}: {result['snippet']} [Source: {result['link']}]")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error performing search: {str(e)}"

class SlackAgent(BaseAgent):
    """Agent for posting messages to Slack channels."""
    
    def __init__(self):
        super().__init__("SlackAgent")
        self.api_token = SLACK_BOT_TOKEN
    
    async def validate_credentials(self):
        return bool(self.api_token)
    
    async def run(self, channel: str, message: str) -> str:
        """Post a message to a Slack channel."""
        logger.info(f"Posting to Slack channel {channel}: {message}")
        
        if not await self.validate_credentials():
            logger.warning("Slack API token not set. Using simulation mode.")
            await asyncio.sleep(1)
            return f"Simulated posting message to Slack channel {channel}"
        
        try:
            # In a real implementation, this would use the Slack API
            # For now, we'll simulate the API call
            await asyncio.sleep(1)
            return f"Message successfully posted to {channel}"
            
        except Exception as e:
            logger.error(f"Slack error: {e}")
            raise Exception(f"Error posting to Slack: {str(e)}")

class EmailAgent(BaseAgent):
    """Agent for sending emails."""
    
    def __init__(self):
        super().__init__("EmailAgent")
        self.api_key = EMAIL_API_KEY
    
    async def validate_credentials(self):
        return bool(self.api_key)
    
    async def run(self, to: str, subject: str, body: str) -> str:
        """Send an email to the specified recipient."""
        logger.info(f"Sending email to {to}: {subject}")
        
        # For demonstration purposes, we'll simulate sending an email
        if not await self.validate_credentials():
            logger.warning("Email API key not set. Using simulation mode.")
            await asyncio.sleep(1)
            return f"Simulated email to {to} with subject: {subject}"
        
        try:
            # In a real implementation, this would use an email API service
            # Simulate API call
            await asyncio.sleep(1)
            return f"Email sent to {to} with subject: {subject}"
            
        except Exception as e:
            logger.error(f"Email error: {e}")
            raise Exception(f"Failed to send email to {to}: {str(e)}")

class CalendarAgent(BaseAgent):
    """Agent for interacting with calendars."""
    
    def __init__(self):
        super().__init__("CalendarAgent")
    
    async def run(self, title: str, start_time: str, end_time: str, 
                 description: str = "", location: str = "", attendees: List[str] = None) -> Dict:
        """Create, update, or delete a calendar event."""
        logger.info(f"Calendar operation: {title} from {start_time} to {end_time}")
        
        # In a real implementation, this would interact with a calendar API
        # For now, we'll simulate the operation
        await asyncio.sleep(1)
        
        event = {
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "location": location,
            "attendees": attendees or [],
            "id": f"event_{hash(title + start_time)}"
        }
        
        return event

class CommunicationAgent(BaseAgent):
    """Agent for making phone calls and sending text messages."""
    
    def __init__(self):
        super().__init__("CommunicationAgent")
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.phone_number = TWILIO_PHONE_NUMBER
    
    async def validate_credentials(self):
        return all([self.account_sid, self.auth_token, self.phone_number])
    
    async def run(self, method: str, recipient: str, message: str) -> str:
        """Make a phone call or send a text message."""
        logger.info(f"{method.capitalize()} to {recipient}: {message}")
        
        # For demonstration purposes, we'll simulate the operation
        if not await self.validate_credentials():
            logger.warning("Twilio credentials not set. Using simulation mode.")
            await asyncio.sleep(1)
            return f"Simulated {method} to {recipient} with message: {message}"
            
        try:
            # In a real implementation, this would use the Twilio API
            # Simulate API call
            await asyncio.sleep(1)
            
            if method.lower() == "text":
                return f"Text message sent to {recipient}"
            elif method.lower() == "call":
                return f"Call initiated to {recipient}"
            else:
                raise ValueError(f"Unsupported communication method: {method}")
                
        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise Exception(f"Failed to {method} {recipient}: {str(e)}")

class WeatherAgent(BaseAgent):
    """Agent for retrieving weather information."""
    
    def __init__(self):
        super().__init__("WeatherAgent")
        self.api_key = WEATHER_API_KEY
    
    async def validate_credentials(self):
        return bool(self.api_key)
    
    async def run(self, location: str, type: str = "current", days: int = 1) -> Dict:
        """Get current weather or forecast for a location."""
        logger.info(f"Getting {type} weather for {location}")
        
        # For demonstration purposes, we'll simulate weather data
        if not await self.validate_credentials():
            logger.warning("Weather API key not set. Using simulation mode.")
            
            if type.lower() == "current":
                return {
                    "location": location,
                    "temperature": 22,
                    "conditions": "Partly Cloudy",
                    "humidity": 65,
                    "wind_speed": 10,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                forecast = []
                for i in range(min(days, 7)):
                    day = datetime.now() + timedelta(days=i)
                    forecast.append({
                        "date": day.strftime("%Y-%m-%d"),
                        "high": 20 + i,
                        "low": 10 + i,
                        "conditions": "Sunny" if i % 2 == 0 else "Cloudy"
                    })
                return {
                    "location": location,
                    "forecast": forecast
                }
        
        try:
            # In a real implementation, this would use a weather API
            # Simulate API call
            await asyncio.sleep(1)
            
            # Return simulated data
            if type.lower() == "current":
                return {
                    "location": location,
                    "temperature": 22,
                    "conditions": "Partly Cloudy",
                    "humidity": 65,
                    "wind_speed": 10,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                forecast = []
                for i in range(min(days, 7)):
                    day = datetime.now() + timedelta(days=i)
                    forecast.append({
                        "date": day.strftime("%Y-%m-%d"),
                        "high": 20 + i,
                        "low": 10 + i,
                        "conditions": "Sunny" if i % 2 == 0 else "Cloudy"
                    })
                return {
                    "location": location,
                    "forecast": forecast
                }
                
        except Exception as e:
            logger.error(f"Weather error: {e}")
            raise Exception(f"Failed to get weather for {location}: {str(e)}")

class BookingAgent(BaseAgent):
    """Agent for booking appointments, reservations, tickets, etc."""
    
    def __init__(self):
        super().__init__("BookingAgent")
        self.api_key = BOOKING_API_KEY
    
    async def validate_credentials(self):
        return bool(self.api_key)
    
    async def run(self, service_type: str, provider: str = None, 
                 details: Dict = None, preferences: Dict = None) -> Dict:
        """Book an appointment, reservation, or ticket."""
        logger.info(f"Booking {service_type} with {provider}: {details}")
        
        details = details or {}
        preferences = preferences or {}
        
        # For demonstration purposes, we'll simulate booking
        if not await self.validate_credentials():
            logger.warning("Booking API key not set. Using simulation mode.")
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            booking_id = f"booking_{hash(str(service_type) + str(provider) + str(datetime.now()))}"
            
            return {
                "booking_id": booking_id,
                "service_type": service_type,
                "provider": provider,
                "status": "confirmed",
                "details": details,
                "confirmation_code": f"CONF{booking_id[-6:]}",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # In a real implementation, this would use booking APIs
            # Simulate API call
            await asyncio.sleep(2)
            
            booking_id = f"booking_{hash(str(service_type) + str(provider) + str(datetime.now()))}"
            
            return {
                "booking_id": booking_id,
                "service_type": service_type,
                "provider": provider,
                "status": "confirmed",
                "details": details,
                "confirmation_code": f"CONF{booking_id[-6:]}",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Booking error: {e}")
            raise Exception(f"Failed to book {service_type} with {provider}: {str(e)}")

# =============================================================================
# TASK ORCHESTRATOR
# =============================================================================

class TaskOrchestrator:
    """Orchestrates the execution of multi-agent tasks based on natural language prompts."""
    
    def __init__(self, task_id: str, prompt: str, ws_manager):
        self.task_id = task_id
        self.prompt = prompt
        self.ws_manager = ws_manager
        self.plan = []
        self.context = {}
        self.start_time = asyncio.get_event_loop().time()
        
        # Initialize agents
        self.agents = {
            "CalendarAgent": CalendarAgent(),
            "CommunicationAgent": CommunicationAgent(),
            "SearchAgent": SearchAgent(),
            "KnowledgeAgent": KnowledgeAgent(),
            "SlackAgent": SlackAgent(),
            "EmailAgent": EmailAgent(),
            "WeatherAgent": WeatherAgent(),
            "BookingAgent": BookingAgent()
        }
        
        # Initialize parsers
        self.parsers = {
            "SlackAgent": SLACK_PARSER_PROMPT_TEMPLATE,
            "EmailAgent": EMAIL_PARSER_PROMPT_TEMPLATE,
            "CalendarAgent": EVENT_PARSER_PROMPT_TEMPLATE,
            "CommunicationAgent": COMMUNICATION_PARSER_PROMPT_TEMPLATE,
            "SearchAgent": SEARCH_QUERY_PARSER_PROMPT_TEMPLATE,
            "WeatherAgent": WEATHER_PARSER_PROMPT_TEMPLATE,
            "BookingAgent": BOOKING_PARSER_PROMPT_TEMPLATE
        }
        
        logger.info(f"Initialized TaskOrchestrator for task {task_id}: {prompt}")

    async def _gemini_request(self, prompt_vars: Dict[str, str], template: str, is_json_output=True):
        """Make a request to the Gemini API with the given prompt template and variables."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        
        headers = {"Content-Type": "application/json"}
        final_prompt = template.format(**prompt_vars)
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": final_prompt}
                    ],
                    "role": "user"
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024,
            }
        }
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    GEMINI_API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=60
                )
                response.raise_for_status()
                response_json = response.json()
                
                content_part = response_json['candidates'][0]['content']['parts'][0]['text']
                
                if is_json_output:
                    # Extract JSON even if it's within markdown code blocks
                    json_text = content_part.strip()
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].strip()
                    
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from Gemini response: {json_text}")
                        raise ValueError("Invalid JSON response from Gemini API")
                
                return content_part.strip()
                
            except requests.RequestException as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to make Gemini API request after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error processing Gemini API response: {e}")
                raise

    async def execute_plan(self):
        """Create and execute a plan based on the user's prompt."""
        try:
            # Log starting task execution
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Starting task execution: {self.prompt}",
                "log_type": "info",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Generate execution plan
            self.plan = await self._gemini_request(
                {
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": self.prompt
                }, 
                PLANNER_PROMPT_TEMPLATE
            )
            
            # Validate plan
            if not self.plan or not isinstance(self.plan, list):
                raise ValueError("Invalid plan generated. Expected a list of steps.")
            
            # Log the plan
            await self.ws_manager.broadcast(json.dumps({
                "type": "plan", 
                "task_id": self.task_id,
                "plan": self.plan
            }))
            
            # Execute each step
            for index, step in enumerate(self.plan):
                # Format any placeholders in the action with context from previous steps
                try:
                    if "{" in step['action']:
                        step['action'] = step['action'].format(**self.context)
                except KeyError as e:
                    logger.warning(f"Missing context key {e} for step: {step['action']}")
                
                # Execute the step
                step_result = await self._execute_step(step, index)
                
                # Update context with result
                if step_result.get('context_updates'):
                    self.context.update(step_result['context_updates'])
            
            # Log task completion
            execution_time = asyncio.get_event_loop().time() - self.start_time
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Task completed in {execution_time:.2f} seconds",
                "log_type": "success",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Send task completion notification
            await self.ws_manager.broadcast(json.dumps({
                "type": "task_complete",
                "task_id": self.task_id,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.exception(f"Error executing plan: {e}")
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Task failed: {str(e)}",
                "log_type": "error",
                "timestamp": datetime.now().isoformat()
            }))

    async def _execute_step(self, step: dict, step_index: int):
        """Execute a single step in the plan."""
        agent_name = step.get('agent', 'UnknownAgent')
        action = step.get('action', 'No action defined')
        description = step.get('description', action)
        
        # Log step starting
        await self.ws_manager.broadcast(json.dumps({
            "type": "status_update",
            "task_id": self.task_id,
            "step_index": step_index,
            "step_action": description,
            "status": "in-progress",
            "timestamp": datetime.now().isoformat()
        }))
        
        await self.ws_manager.broadcast(json.dumps({
            "type": "log",
            "task_id": self.task_id,
            "agent": agent_name,
            "message": f"Starting: {description}",
            "log_type": "info",
            "timestamp": datetime.now().isoformat()
        }))
        
        execution_result = {
            "success": True,
            "message": f"Completed: {description}",
            "context_updates": {}
        }
        
        try:
            # Check if the agent exists
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            agent = self.agents[agent_name]
            
            # Parse the action according to agent type
            if agent_name in self.parsers:
                # Get the appropriate parser template
                parser_template = self.parsers[agent_name]
                
                # For SearchAgent, we don't need full JSON parsing
                if agent_name == "SearchAgent":
                    query = await self._gemini_request(
                        {"action_text": action}, 
                        parser_template, 
                        is_json_output=False
                    )
                    result = await agent.run(query)
                    execution_result["message"] = f"Found results for: '{query}'"
                    execution_result["context_updates"]["search_result"] = result
                    
                else:
                    # Parse the action details
                    parsed_details = await self._gemini_request(
                        {"action_text": action},
                        parser_template
                    )
                    
                    # Execute with the parsed details
                    result = await agent.run(**parsed_details)
                    
                    # Update context with result
                    if agent_name == "KnowledgeAgent":
                        execution_result["context_updates"]["knowledge_answer"] = result
                    elif agent_name == "WeatherAgent":
                        execution_result["context_updates"]["weather_info"] = result
                    elif agent_name == "SlackAgent":
                        execution_result["message"] = f"Message posted to {parsed_details['channel']}"
                    elif agent_name == "EmailAgent":
                        execution_result["message"] = f"Email sent to {parsed_details['to']}"
                    elif agent_name == "CalendarAgent":
                        execution_result["context_updates"]["calendar_event"] = result
                        execution_result["message"] = f"Calendar event '{parsed_details['title']}' processed"
                    elif agent_name == "BookingAgent":
                        execution_result["context_updates"]["booking_details"] = result
                        execution_result["message"] = f"Booking completed for {parsed_details['service_type']}"
                    
            else:
                # For agents without parsers, just pass the action directly
                result = await agent.run(action)
                execution_result["context_updates"][f"{agent_name.lower()}_result"] = result
            
        except Exception as e:
            logger.exception(f"Error executing step {step_index}: {e}")
            execution_result = {
                "success": False,
                "message": f"Failed: {str(e)}",
                "context_updates": {}
            }
        
        # Log step completion
        status = "completed" if execution_result["success"] else "failed"
        log_type = "info" if execution_result["success"] else "error"
        
        await self.ws_manager.broadcast(json.dumps({
            "type": "status_update",
            "task_id": self.task_id,
            "step_index": step_index,
            "step_action": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }))
        
        await self.ws_manager.broadcast(json.dumps({
            "type": "log",
            "task_id": self.task_id,
            "agent": agent_name,
            "message": execution_result["message"],
            "log_type": log_type,
            "timestamp": datetime.now().isoformat()
        }))
        
        return execution_result

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Automation System</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .task-card {
            transition: all 0.3s ease;
        }
        .task-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .log-container {
            max-height: 300px;
            overflow-y: auto;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-10">
            <div class="flex justify-between items-center">
                <h1 class="text-3xl font-bold text-gray-800">Task Automation System</h1>
                <div class="flex items-center">
                    <div id="connectionStatus" class="flex items-center mr-4">
                        <span class="h-3 w-3 rounded-full bg-gray-400 mr-2"></span>
                        <span class="text-sm text-gray-600">Connecting...</span>
                    </div>
                </div>
            </div>
            <p class="text-gray-600 mt-2">Describe your task in natural language and let our AI agents handle it for you.</p>
        </header>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Task Input Section -->
            <div class="lg:col-span-2 bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">What do you want to accomplish?</h2>
                <div class="mb-4">
                    <textarea id="taskInput" rows="4" class="w-full px-3 py-2 text-gray-700 border rounded-lg focus:outline-none focus:shadow-outline" placeholder="Example: Find doctors near me who accept Blue Cross insurance, sort by rating, call the highest-rated one for an appointment next Tuesday, and add it to my calendar."></textarea>
                </div>
                <div class="flex justify-end">
                    <button id="submitTaskBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg focus:outline-none focus:shadow-outline transition duration-300">
                        Execute Task
                    </button>
                </div>
                
                <!-- Current Task Status -->
                <div id="currentTaskSection" class="mt-8 hidden">
                    <h3 class="text-lg font-semibold text-gray-800 mb-3">Current Task</h3>
                    <div id="currentTaskCard" class="border border-gray-200 rounded-lg p-4">
                        <p id="currentTaskDescription" class="text-gray-700 mb-3"></p>
                        <div class="mb-4">
                            <h4 class="text-sm font-medium text-gray-600 mb-2">Execution Plan:</h4>
                            <ul id="executionPlanList" class="list-disc pl-5 text-sm text-gray-600"></ul>
                        </div>
                        <div>
                            <h4 class="text-sm font-medium text-gray-600 mb-2">Progress:</h4>
                            <div id="progressSteps" class="space-y-2"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- System Logs Section -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">System Logs</h2>
                <div id="logContainer" class="log-container space-y-2 p-3 bg-gray-50 rounded-lg"></div>
            </div>
        </div>
        
        <!-- Task History Section -->
        <div class="mt-10">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Task History</h2>
            <div id="taskHistoryContainer" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <!-- Task history cards will be added here -->
                <div class="text-gray-500 text-center py-8">No tasks completed yet</div>
            </div>
        </div>
    </div>
    
    <script>
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
    </script>
    
    <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
</body>
</html>"""

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main HTML page."""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest):
    """Create a new task from a natural language prompt."""
    global task_counter
    task_counter += 1
    task_id = f"task_{task_counter}"
    
    logger.info(f"Received task: {task_request.prompt} from user: {task_request.user_id}")
    
    # Create task entry
    tasks_db[task_id] = {
        "prompt": task_request.prompt,
        "user_id": task_request.user_id,
        "status": "scheduled",
        "created_at": asyncio.get_event_loop().time(),
        "updated_at": asyncio.get_event_loop().time(),
    }
    
    # Create and execute orchestrator
    orchestrator = TaskOrchestrator(task_id, task_request.prompt, ws_manager)
    asyncio.create_task(orchestrator.execute_plan())
    
    return TaskResponse(
        task_id=task_id,
        status="scheduled",
        message="Task has been scheduled for execution"
    )

@app.get("/api/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str):
    """Get details of a specific task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]

@app.get("/api/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(user_id: Optional[str] = None):
    """List all tasks, optionally filtered by user_id."""
    if user_id:
        return [task for task_id, task in tasks_db.items() if task["user_id"] == user_id]
    return list(tasks_db.values())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for client connections."""
    await ws_manager.connect(websocket, client_id)
    logger.info(f"WebSocket connection established with client {client_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.info(f"Received message from client {client_id}: {message}")
                
                # Handle different message types here
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong", 
                        "timestamp": json.loads(data).get("timestamp")
                    }))
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
                
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        logger.info(f"WebSocket connection closed for client {client_id}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_knowledge_base():
    """Create a sample knowledge base directory if it doesn't exist."""
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        with open(os.path.join("knowledge_base", "sample.txt"), "w") as f:
            f.write("This is sample knowledge. Replace with actual information about your organization, products, or domain.")

if __name__ == "__main__":
    print("="*80)
    print("Starting Multi-Agent Task Automation System")
    print("Open your browser and navigate to http://localhost:8000")
    print("Press CTRL+C to stop the server")
    print("="*80)
    
    # Create knowledge base directory
    create_knowledge_base()
    
    try:
        # Start the FastAPI application
        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
