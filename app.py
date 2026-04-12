"""
AI Foresight Scanner - Lightweight Web Version
A simple, fast version that works within web hosting limits.
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps

import httpx
import anthropic
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))

# Path to the user database file (used as fallback when DATABASE_URL is not set)
USER_DB_PATH = os.path.join(os.path.dirname(__file__), 'users.json')
# Admin password for the /admin management panel (set via env var)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')
# PostgreSQL connection URL (set this env var on the hosting platform)
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# API clients
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
BRAVE_API_KEY = os.environ.get('BRAVE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

MODEL = 'claude-sonnet-4-20250514'

# --- Input validation constants ---
MAX_TOPIC_LENGTH = 200
# Patterns that suggest sensitive data
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}')
# Policy / claim reference patterns (e.g., POL-123456, CLM-2024-001, etc.)
POLICY_CLAIM_PATTERN = re.compile(
    r'\b(?:POL|CLM|REF|INV|ACC|CLAIM|POLICY|CERT)[\s\-#]?\d{3,}',
    re.IGNORECASE
)
MULTI_PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')


def validate_topic_input(topic: str) -> str | None:
    """Validate topic input and return error message if invalid, else None."""
    if len(topic) > MAX_TOPIC_LENGTH:
        return f'Input too long. Please keep your topic under {MAX_TOPIC_LENGTH} characters.'

    if EMAIL_PATTERN.search(topic):
        return 'Input appears to contain an email address. Please enter only a general topic using publicly available terms.'

    if PHONE_PATTERN.search(topic):
        return 'Input appears to contain a phone number. Please enter only a general topic using publicly available terms.'

    if POLICY_CLAIM_PATTERN.search(topic):
        return 'Input appears to contain a policy, claim, or reference number. Please enter only a general topic using publicly available terms.'

    if MULTI_PARAGRAPH_PATTERN.search(topic):
        return 'Multi-paragraph text is not accepted. Please enter a short topic description (one line).'

    return None


def _db_url() -> str:
    """Return a psycopg2-compatible DB URL (postgresql:// scheme)."""
    url = DATABASE_URL
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return url


def _init_db() -> None:
    """Create the user_store table and seed row if they don't exist."""
    if not DATABASE_URL:
        _apply_seed_users()
        return
    import psycopg2
    conn = psycopg2.connect(_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_store (
                    id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL DEFAULT '{}'
                )
            """)
            cur.execute("""
                INSERT INTO user_store (id, data) VALUES (1, '{}')
                ON CONFLICT (id) DO NOTHING
            """)
        conn.commit()
    finally:
        conn.close()
    _apply_seed_users()


def _apply_seed_users() -> None:
    """Merge users from the SEED_USERS env var into the active store.

    Set SEED_USERS to a JSON object mapping email -> user record, e.g.:
        {"alice@example.com": {"name": "Alice", "password_hash": "...", "enabled": true}}

    Existing records are never overwritten — only missing users are added.
    This ensures accounts survive redeployments on platforms with ephemeral
    filesystems even when DATABASE_URL is not configured.
    """
    raw = os.environ.get('SEED_USERS', '').strip()
    if not raw:
        return
    try:
        seed = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(seed, dict):
        return
    users = load_users()
    changed = False
    for email, record in seed.items():
        email = email.lower()
        if email not in users:
            users[email] = record
            changed = True
    if changed:
        save_users(users)


def load_users() -> dict:
    """Load users from PostgreSQL when DATABASE_URL is set, else from users.json."""
    if DATABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(_db_url())
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT data FROM user_store WHERE id = 1")
                    row = cur.fetchone()
                    return json.loads(row[0]) if row else {}
            finally:
                conn.close()
        except Exception:
            pass

    if not os.path.exists(USER_DB_PATH):
        return {}
    try:
        with open(USER_DB_PATH) as f:
            return json.load(f).get('users', {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users: dict) -> None:
    """Persist users to PostgreSQL when DATABASE_URL is set, else to users.json."""
    if DATABASE_URL:
        import psycopg2
        conn = psycopg2.connect(_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_store SET data = %s WHERE id = 1",
                    (json.dumps(users),)
                )
            conn.commit()
        finally:
            conn.close()
        return

    with open(USER_DB_PATH, 'w') as f:
        json.dump({'users': users}, f, indent=2)


def verify_user(username: str, password: str) -> bool:
    """Return True if credentials are valid and the user account is enabled."""
    users = load_users()
    user = users.get(username.lower())
    if not user or not user.get('enabled', True):
        return False
    return check_password_hash(user['password_hash'], password)


def require_auth(f):
    """Decorator to require user authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('username'):
            # Return JSON error for AJAX requests instead of redirecting to HTML
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
               request.content_type == 'multipart/form-data' or \
               request.accept_mimetypes.best == 'application/json':
                return jsonify({'error': 'Session expired. Please refresh the page and log in again.'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def require_admin(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def search_web(query: str, num_results: int = 10) -> list:
    """Search the web using Serper or Brave API."""
    results = []

    if SERPER_API_KEY:
        # Use Serper
        response = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY},
            json={"q": query, "num": num_results},
            timeout=15.0
        )
        if response.status_code == 200:
            data = response.json()
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })

    elif BRAVE_API_KEY:
        # Use Brave
        response = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": num_results},
            timeout=15.0
        )
        if response.status_code == 200:
            data = response.json()
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "link": item.get("url", "")
                })

    return results


def deduplicate_results(results: list, key: str = 'link') -> list:
    """Return results with duplicates removed, preserving order."""
    seen = set()
    unique = []
    for r in results:
        val = r.get(key)
        if val and val not in seen:
            seen.add(val)
            unique.append(r)
    return unique


def parallel_search(queries: list[str], num_results: int = 10) -> list:
    """Run multiple search_web calls concurrently and return combined, deduplicated results."""
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        futures = [executor.submit(search_web, q, num_results) for q in queries]
        all_results = []
        for f in futures:
            all_results.extend(f.result())
    return deduplicate_results(all_results)


def extract_json_object(response_text: str) -> str:
    """Strip markdown fences and extract the first JSON object from a Claude response."""
    text = response_text.strip()
    if '```' in text:
        parts = text.split('```')
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith('json'):
            inner = inner[4:]
        text = inner.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


def analyze_with_claude(topic: str, search_results: list) -> dict:
    """Use Claude to analyze search results and identify trends."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=240.0)

    # Format search results for the prompt
    sources_text = ""
    for i, result in enumerate(search_results, 1):
        sources_text += f"\n{i}. {result['title']}\n   {result['snippet']}\n   Source: {result['link']}\n"

    prompt = f"""You are a strategic foresight analyst following a systematic methodology for identifying phenomena. Based on the search results below about "{topic}", identify exactly 20 key phenomena.

PHENOMENON CRITERIA - Each phenomenon must meet ALL of these:
1. It must have a significant impact on several industries in the future.
2. Its potential impact is informed by the available evidence (not speculation alone).
3. It must be covered in several trustworthy publications for verification purposes (wild cards and weak signals are treated more flexibly here).
4. It must have a direction: either getting stronger, broader, deeper, or weaker, or merging with other phenomena. General themes like "Use of fossil fuels" or "Sharing economy" alone are NOT phenomena.
5. It must have a sufficiently independent and robust core description that can be verified.

COLOUR-CODED SIGNAL TYPES - You MUST include at least one of each:
- "Strengthening" (GREEN): The issue is becoming more common or acute during the given timeframe. Most of its change potential is still ahead.
- "Weakening" (BLUE): The issue is becoming more unusual. During the given timeframe, most of its change potential or value has already occurred.
- "Established" (PURPLE): The issue has stabilised in its development. It has future relevance, but there is no indication it will significantly strengthen or weaken within the given timeframe.
- "Weak Signal" (GREY): A small emerging issue. At the given timeframe, it is still hard to say whether it will become a trend or fade away without substantial impact.
- "Wild Card" (RED): A possible but not probable event or change. Early information about a potential emerging risk or opportunity. Probability within the given timeframe is between 5% to 30%.

TIMING: Each phenomenon has an expert-assessed timeframe within which it is anticipated to either accelerate or decline, determined using S-Curve Analysis and Trend Impact Analysis. Use one of:
- "Near-term (0-5 years)"
- "Mid-term (5-10 years)"
- "Long-term (10-20 years)"
IMPORTANT: Do NOT specify timing for Weak Signals. Weak signals are observations of a potential change just beginning to form, and there isn't enough data to assess their possible development paths. For weak signals, set timing to null.

THEME TAGS: Each phenomenon MUST be assigned exactly one category from these four fixed categories: "Strategic", "Regulatory", "Operational", "Financial". This category determines the radar quadrant. The theme_tags array should contain only this single category.

WRITING STYLE - VALUE RATIONALITY:
- Avoid dichotomous good-bad appraisals. Present descriptions in a neutral manner.
- Write descriptions as versatile and multifaceted analyses, originating from one single set of values but applicable to multiple perspectives.
- The summary should help the reader recognise the point of view the text represents, the formulation used, and the potential way to use it.
- Phenomenon descriptions are not truths carved in stone; they are analyses from one set of values that can be interpreted from multiple angles.

SOURCE RELIABILITY:
- The core of each phenomenon must be backed up by reliable sources.
- Prioritise: peer-reviewed scientific journals (Nature, Science), self-evidently proper scientific journals, Reuters, CNN, BBC, Financial Times, The Guardian, Wired, Scientific American.
- Also consider publications by universities, international research organisations (World Economic Forum, OECD).
- Internet-based, mainly ad-supported and freelance driven news distribution sites such as Popular Mechanics or Interesting Engineering are considered sufficiently reliable to be used as sources.

SEARCH RESULTS:
{sources_text}

For each phenomenon, provide:
1. **Title**: A clear, concise name explaining the core of the phenomenon in a few words. Titles can be general (e.g., "On-Demand Services"), industry-specific (e.g., "Robotics in Healthcare"), or for wild cards, a mini-sentence describing a potential future state (e.g., "Knowledge Behind Paywall").
2. **Theme Tags**: Exactly one category from: "Strategic", "Regulatory", "Operational", "Financial" (e.g., ["Strategic"])
3. **Type**: One of: "Strengthening", "Weakening", "Established", "Weak Signal", or "Wild Card"
4. **Timing**: "Near-term (0-5 years)", "Mid-term (5-10 years)", or "Long-term (10-20 years)". Set to null for Weak Signals.
5. **Summary**: A single paragraph synopsis explaining the core of the phenomenon, its current situation, and its most likely future development path and impacts.
6. **Background**: 1-2 sentences outlining the phenomenon's history, relevance, and current state.
7. **Impact**: 1-2 sentences describing the phenomenon's potential impacts with prominent case examples.
8. **Additional Information**: 1-3 additional source references (statistics, news articles, journal articles, product releases, or opinion pieces) that provide further context. Each entry should include the article title, the source URL, and a brief description of what the source covers. Format each entry as: "Article Title (URL): description".
9. **Source Confidence**: Assess the overall quality of sources backing this phenomenon. Return "High" (backed by multiple tier-1 sources such as Reuters, BBC, peer-reviewed journals, WEF, OECD), "Medium" (supported by a credible mix of sources), or "Low" (primarily weaker sources, single references, or speculation-heavy content).
10. **Emerging Risks**: 2-3 sentences identifying potential risks that could emerge from this phenomenon over the coming years — risks that are not yet fully formed but that this phenomenon could give rise to.
11. **Insurance Impact**: 2-3 sentences explaining the specific implications of this phenomenon for insurance brokers and their clients — covering which lines of coverage are most affected, potential claims exposure, and any client advisory actions brokers should consider.

Format your response as a JSON array like this:
[
  {{
    "title": "Example Trend",
    "theme_tags": ["Strategic"],
    "type": "Strengthening",
    "timing": "Near-term (0-5 years)",
    "summary": "Synopsis paragraph here...",
    "background": "History, relevance, and current state here...",
    "impact": "Potential impacts with case examples here...",
    "additional_information": ["Article Title (https://example.com/article): description of what it covers", "Another Article (https://example.com/article2): description"],
    "source_confidence": "High",
    "emerging_risks": "Description of potential risks not yet fully formed that this phenomenon could give rise to over the coming years...",
    "insurance_impact": "Implications for insurance brokers and clients, including which lines of coverage are most affected, potential claims exposure, and advisory actions brokers should consider..."
  }}
]

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the response
    response_text = response.content[0].text.strip()

    # Try to extract JSON from the response
    try:
        # Remove markdown code blocks if present
        if "```" in response_text:
            # Extract content between first pair of triple backticks
            parts = response_text.split("```")
            if len(parts) >= 3:
                inner = parts[1]
            else:
                inner = parts[1] if len(parts) > 1 else response_text
            if inner.startswith("json"):
                inner = inner[4:]
            response_text = inner.strip()

        # Find the JSON array boundaries
        start = response_text.find('[')
        if start != -1:
            end = response_text.rfind(']')
            if end != -1:
                response_text = response_text[start:end + 1]

        phenomena = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to salvage truncated JSON by closing open structures
        try:
            # Find last complete object (ends with })
            last_complete = response_text.rfind('}')
            if last_complete != -1:
                truncated = response_text[:last_complete + 1]
                # Ensure it starts with [ and ends properly
                if '[' in truncated:
                    truncated = truncated[truncated.find('['):]
                    if not truncated.endswith(']'):
                        truncated += ']'
                    phenomena = json.loads(truncated)
                else:
                    raise json.JSONDecodeError("No array found", "", 0)
            else:
                raise json.JSONDecodeError("No objects found", "", 0)
        except json.JSONDecodeError:
            phenomena = [{
                "title": "Analysis Error",
                "type": "Note",
                "timing": None,
                "theme_tags": [],
                "summary": "The AI response could not be parsed. This may be due to a temporary issue. Please try scanning again.",
                "background": "",
                "impact": "",
                "additional_information": []
            }]

    return {
        "topic": topic,
        "phenomena": phenomena,
        "sources": search_results
    }


def generate_executive_summary(topic: str, phenomena: list) -> dict:
    """Generate a 3-sentence executive brief from the identified phenomena."""
    if not phenomena:
        return {"dominant_theme": "", "most_urgent": "", "biggest_wildcard": ""}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

    phenomena_text = "\n".join([
        f"- [{p.get('type', '')}] {p.get('title', '')}: {p.get('summary', '')[:180]}"
        for p in phenomena[:20]
    ])

    prompt = f"""You are a strategic foresight analyst. Based on the {len(phenomena)} phenomena identified for the topic "{topic}", write a concise executive brief with exactly 3 sentences covering:

1. The DOMINANT THEME: What overarching pattern or direction connects the majority of these phenomena?
2. The MOST URGENT SIGNAL: Which single phenomenon demands the most immediate attention, and why?
3. The BIGGEST WILDCARD: What is the most unexpected or potentially disruptive phenomenon, and what makes it unpredictable?

PHENOMENA:
{phenomena_text}

Return ONLY a JSON object with exactly these three fields (no markdown, no preamble):
{{"dominant_theme": "One sentence about the dominant theme.", "most_urgent": "One sentence about the most urgent signal.", "biggest_wildcard": "One sentence about the biggest wildcard."}}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(extract_json_object(response.content[0].text))
    except Exception:
        return {"dominant_theme": "", "most_urgent": "", "biggest_wildcard": ""}


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Username + password login."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        if username and verify_user(username, password):
            session['username'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/change-password', methods=['GET', 'POST'])
@require_auth
def change_password():
    """Allow the logged-in user to change their own password."""
    if request.method == 'POST':
        current = request.form.get('current_password', '')
        new_pw = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        username = session['username']

        if not verify_user(username, current):
            return render_template('change_password.html', error='Current password is incorrect.')
        if len(new_pw) < 8:
            return render_template('change_password.html', error='New password must be at least 8 characters.')
        if new_pw != confirm:
            return render_template('change_password.html', error='New passwords do not match.')

        users = load_users()
        users[username]['password_hash'] = generate_password_hash(new_pw)
        save_users(users)
        return render_template('change_password.html', success=True)

    return render_template('change_password.html')


# ---------------------------------------------------------------------------
# Admin – user management
# ---------------------------------------------------------------------------

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin password gate."""
    if not ADMIN_PASSWORD:
        return 'Admin access is disabled. Set the ADMIN_PASSWORD environment variable to enable it.', 403
    if request.method == 'POST':
        if request.form.get('password', '') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin'))
        return render_template('admin_login.html', error='Incorrect admin password.')
    return render_template('admin_login.html', error=None)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
@require_admin
def admin():
    """User management dashboard."""
    users = load_users()
    return render_template('admin.html', users=users)


@app.route('/admin/users/add', methods=['POST'])
@require_admin
def admin_add_user():
    """Create a new user account."""
    username = request.form.get('username', '').strip().lower()
    name = request.form.get('name', '').strip()
    password = request.form.get('password', '')

    if not username or not password:
        flash('Username and password are required.', 'error')
        return redirect(url_for('admin'))

    if not re.match(r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$', username):
        flash('Please enter a valid email address as the username.', 'error')
        return redirect(url_for('admin'))

    users = load_users()
    if username in users:
        flash(f'User "{username}" already exists.', 'error')
        return redirect(url_for('admin'))

    users[username] = {
        'name': name or username,
        'password_hash': generate_password_hash(password),
        'enabled': True,
        'created': datetime.now(timezone.utc).isoformat(),
    }
    save_users(users)
    flash(f'User "{username}" created successfully.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<username>/toggle', methods=['POST'])
@require_admin
def admin_toggle_user(username):
    """Enable or disable a user account."""
    users = load_users()
    if username in users:
        users[username]['enabled'] = not users[username].get('enabled', True)
        save_users(users)
        state = 'enabled' if users[username]['enabled'] else 'disabled'
        flash(f'User "{username}" {state}.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<username>/delete', methods=['POST'])
@require_admin
def admin_delete_user(username):
    """Permanently delete a user account."""
    users = load_users()
    if username in users:
        users.pop(username)
        save_users(users)
        flash(f'User "{username}" deleted.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/users/<username>/reset', methods=['POST'])
@require_admin
def admin_reset_password(username):
    """Reset a user's password."""
    new_password = request.form.get('password', '')
    if not new_password:
        flash('New password cannot be empty.', 'error')
        return redirect(url_for('admin'))
    users = load_users()
    if username in users:
        users[username]['password_hash'] = generate_password_hash(new_password)
        save_users(users)
        flash(f'Password for "{username}" updated.', 'success')
    return redirect(url_for('admin'))


@app.route('/')
@require_auth
def home():
    """Show the main page."""
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
@require_auth
def scan_topic():
    """Handle the scan request."""
    try:
        topic = request.form.get('topic', '').strip()
        attestation = request.form.get('attestation', '')

        if not topic:
            return jsonify({'error': 'Please enter a topic to scan'}), 400

        # Require attestation
        if attestation != 'confirmed':
            return jsonify({'error': 'You must confirm the public-data attestation before submitting.'}), 400

        # Log attestation timestamp
        attestation_time = datetime.now(timezone.utc).isoformat()
        app.logger.info(f"Public-data attestation confirmed at {attestation_time} for topic: {topic[:50]}")

        # Validate input against sensitive patterns
        validation_error = validate_topic_input(topic)
        if validation_error:
            return jsonify({'error': validation_error}), 400

        # Check API keys
        if not ANTHROPIC_API_KEY:
            return jsonify({'error': 'Anthropic API key not configured'}), 500
        if not SERPER_API_KEY and not BRAVE_API_KEY:
            return jsonify({'error': 'No search API key configured (need SERPER_API_KEY or BRAVE_API_KEY)'}), 500

        # Step 1: Search the web (5 targeted queries, run in parallel)
        search_queries = [
            f"{topic} trends 2024 2025",
            f"{topic} future predictions emerging",
            f"{topic} risk factors",
            f"{topic} regulatory changes",
            f"{topic} industry disruption"
        ]

        unique_results = parallel_search(search_queries, num_results=10)

        if not unique_results:
            return jsonify({'error': 'No search results found. Please try a different topic.'}), 400

        # Step 2: Analyze with Claude (single API call)
        analysis_sources = unique_results[:20]
        analysis = analyze_with_claude(topic, analysis_sources)

        # Step 3: Generate executive summary (second, lightweight API call)
        executive_summary = generate_executive_summary(topic, analysis['phenomena'])

        return jsonify({
            'success': True,
            'topic': analysis['topic'],
            'phenomena_count': len(analysis['phenomena']),
            'phenomena': analysis['phenomena'],
            'sources': unique_results,
            'executive_summary': executive_summary,
            'attestation_timestamp': attestation_time
        })

    except anthropic.APIError as e:
        return jsonify({'error': f'AI API error: {str(e)}'}), 500
    except httpx.TimeoutException:
        return jsonify({'error': 'Search timed out. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


REGION_NEWS_QUERIES = {
    'world': 'major world news today international headlines',
    'us': 'United States news today Washington politics economy',
    'europe': 'Europe EU news today',
    'middle-east': 'Middle East news today',
    'africa': 'Africa news today',
    'latin-america': 'Latin America South America news today',
    'asia-pacific': 'Asia Pacific China Japan India news today',
}

REGION_INTEL_QUERIES = {
    'us': [
        'United States geopolitical security risk today',
        'North America military diplomatic tensions 2025',
        'US economy financial risk news today',
    ],
    'europe': [
        'Europe EU security geopolitical crisis today',
        'European military NATO tensions 2025',
        'Europe economic financial risk news today',
    ],
    'middle-east': [
        'Middle East conflict security crisis today',
        'Israel Iran Gulf tensions military 2025',
        'Middle East geopolitical risk economy today',
    ],
    'africa': [
        'Africa security conflict crisis today',
        'Sub-Saharan Africa geopolitical risk 2025',
        'Africa economic instability news today',
    ],
    'latin-america': [
        'Latin America South America security risk today',
        'Latin America political instability crisis 2025',
        'South America economic risk news today',
    ],
    'asia-pacific': [
        'Asia Pacific Indo-Pacific security tensions today',
        'China Taiwan South China Sea military 2025',
        'Asia Pacific economic geopolitical risk today',
    ],
}

REGION_DISPLAY_NAMES = {
    'us': 'North America',
    'europe': 'Europe',
    'middle-east': 'Middle East',
    'africa': 'Africa',
    'latin-america': 'Latin America',
    'asia-pacific': 'Indo-Pacific',
}


@app.route('/api/intelligence')
@require_auth
def get_intelligence():
    """Generate AI world brief, strategic posture, and strategic risk overview."""
    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500
    if not SERPER_API_KEY and not BRAVE_API_KEY:
        return jsonify({'error': 'No search API key configured'}), 500

    VALID_FILTER_TYPES = {'all', 'geopolitical', 'macroeconomic', 'regulatory', 'climate_natcat'}
    filter_type = request.args.get('filter_type', 'all')
    if filter_type not in VALID_FILTER_TYPES:
        filter_type = 'all'

    FILTER_QUERIES = {
        'all': [
            'global security geopolitical crisis news today',
            'international military conflict tensions 2025',
            'world economic financial market risk today',
            'major breaking news international today',
        ],
        'geopolitical': [
            'global military conflict tensions today',
            'geopolitical crisis diplomacy sanctions 2025',
            'war conflict flashpoints international security today',
            'great power rivalry US China Russia today',
        ],
        'macroeconomic': [
            'global economy recession risk today',
            'financial markets instability inflation 2025',
            'trade wars tariffs economic disruption today',
            'central bank policy interest rates global economy today',
        ],
        'regulatory': [
            'global regulatory policy changes 2025',
            'international compliance legislation financial regulation today',
            'government policy shifts sanctions regulation today',
            'ESG regulation data privacy antitrust policy 2025',
        ],
        'climate_natcat': [
            'extreme weather events natural disasters today 2025',
            'climate risk flood earthquake wildfire hurricane today',
            'natural catastrophe high risk regions 2025',
            'climate change physical risk insurance exposure today',
        ],
    }

    FILTER_FOCUS = {
        'all': 'Focus on the most significant global risks across all categories.',
        'geopolitical': 'Focus exclusively on geopolitical risks — military, diplomatic, conflict, and power dynamics.',
        'macroeconomic': 'Focus exclusively on macroeconomic risks — markets, trade, inflation, financial stability, and economic policy.',
        'regulatory': 'Focus exclusively on regulatory risks — policy changes, legislation, compliance shifts, and government interventions.',
        'climate_natcat': 'Focus exclusively on climate and natural catastrophe risks — extreme weather, natural disasters, high-risk regions for floods, earthquakes, wildfires, and hurricanes, and physical climate risk exposure.',
    }

    try:
        unique_results = parallel_search(FILTER_QUERIES[filter_type], num_results=7)

        news_text = '\n'.join(
            f"- {r['title']}: {r['snippet']}"
            for r in unique_results[:24]
        )

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)

        prompt = f"""You are a strategic intelligence analyst. Based on these current news headlines, provide a structured assessment.

NEWS HEADLINES:
{news_text}

{FILTER_FOCUS[filter_type]}

Provide a JSON response with exactly these three sections:
{{
  "world_brief": "2-3 sentence authoritative intelligence briefing summarising the most significant global developments right now.",
  "strategic_posture": {{
    "overall": "ELEVATED",
    "summary": "2-3 sentences assessing the current global strategic posture - key military, diplomatic, and power dynamics.",
    "theaters": [
      {{"region": "North America", "status": "NORMAL", "note": "your assessment here"}},
      {{"region": "Europe", "status": "ELEVATED", "note": "your assessment here"}},
      {{"region": "Middle East", "status": "ELEVATED", "note": "your assessment here"}},
      {{"region": "Indo-Pacific", "status": "HEIGHTENED", "note": "your assessment here"}},
      {{"region": "Africa", "status": "NORMAL", "note": "your assessment here"}},
      {{"region": "Latin America", "status": "NORMAL", "note": "your assessment here"}}
    ]
  }},
  "strategic_risk": {{
    "score": 62,
    "trend": "STABLE",
    "top_risks": [
      {{"title": "...", "severity": "HIGH", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "HIGH", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "MEDIUM", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "MEDIUM", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "LOW", "description": "...", "insurance_implications": "..."}}
    ],
    "summary": "1-2 sentences on the overall risk environment."
  }}
}}

Rules:
- "overall" must be one of: NORMAL, ELEVATED, HEIGHTENED, CRITICAL
- Each theater "status" must be one of: NORMAL, ELEVATED, HEIGHTENED, CRITICAL
- "trend" must be one of: ESCALATING, STABLE, DE-ESCALATING
- Each risk "severity" must be one of: HIGH, MEDIUM, LOW
- "score" must be a number 0-100 reflecting current global risk level
- Each risk "insurance_implications" must be 1-2 sentences explaining what that specific risk means for an insurance broker and their clients — covering relevant lines of coverage, potential claims exposure, or client advisory actions
- Base all assessments on the provided headlines; replace all "..." placeholders with real content

Return ONLY valid JSON matching the structure above, no markdown or extra text."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{'role': 'user', 'content': prompt}]
        )

        intelligence = json.loads(extract_json_object(response.content[0].text))
        intelligence['timestamp'] = datetime.now(timezone.utc).isoformat()
        intelligence['filter_type'] = filter_type

        return jsonify({'success': True, 'intelligence': intelligence})

    except anthropic.APIError as e:
        return jsonify({'error': f'AI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/regional-intelligence')
@require_auth
def get_regional_intelligence():
    """Generate AI intelligence brief focused on a specific region."""
    region = request.args.get('region', '').lower().strip()

    if region not in REGION_INTEL_QUERIES:
        return jsonify({'error': 'Invalid region'}), 400

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500
    if not SERPER_API_KEY and not BRAVE_API_KEY:
        return jsonify({'error': 'No search API key configured'}), 500

    region_name = REGION_DISPLAY_NAMES.get(region, region)

    try:
        unique_results = parallel_search(REGION_INTEL_QUERIES[region], num_results=7)

        news_text = '\n'.join(
            f"- {r['title']}: {r['snippet']}"
            for r in unique_results[:20]
        )

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)

        prompt = f"""You are a strategic intelligence analyst specialising in {region_name}. Based on these current news headlines, provide a structured regional assessment focused exclusively on {region_name}.

NEWS HEADLINES:
{news_text}

Provide a JSON response with exactly these three sections:
{{
  "world_brief": "2-3 sentence authoritative intelligence briefing on the most significant developments in {region_name} right now.",
  "strategic_posture": {{
    "overall": "ELEVATED",
    "summary": "2-3 sentences assessing the current strategic posture in {region_name} — key military, diplomatic, and power dynamics specific to this region.",
    "theaters": [
      {{"region": "{region_name}", "status": "ELEVATED", "note": "your detailed regional assessment here"}},
      {{"region": "Key Sub-region or Actor", "status": "NORMAL", "note": "your assessment here"}},
      {{"region": "Key Sub-region or Actor", "status": "NORMAL", "note": "your assessment here"}}
    ]
  }},
  "strategic_risk": {{
    "score": 62,
    "trend": "STABLE",
    "top_risks": [
      {{"title": "...", "severity": "HIGH", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "HIGH", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "MEDIUM", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "MEDIUM", "description": "...", "insurance_implications": "..."}},
      {{"title": "...", "severity": "LOW", "description": "...", "insurance_implications": "..."}}
    ],
    "summary": "1-2 sentences on the overall risk environment specifically in {region_name}."
  }}
}}

Rules:
- "overall" must be one of: NORMAL, ELEVATED, HEIGHTENED, CRITICAL
- Each theater "status" must be one of: NORMAL, ELEVATED, HEIGHTENED, CRITICAL
- "trend" must be one of: ESCALATING, STABLE, DE-ESCALATING
- Each risk "severity" must be one of: HIGH, MEDIUM, LOW
- "score" must be a number 0-100 reflecting the regional risk level
- Each risk "insurance_implications" must be 1-2 sentences explaining what that specific risk means for an insurance broker and their clients — covering relevant lines of coverage, potential claims exposure, or client advisory actions
- All content must be specifically about {region_name}, not global
- Base all assessments on the provided headlines; replace all "..." placeholders with real content
- Use real sub-regions, countries, or key actors relevant to {region_name} for the theaters list

Return ONLY valid JSON matching the structure above, no markdown or extra text."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{'role': 'user', 'content': prompt}]
        )

        intelligence = json.loads(extract_json_object(response.content[0].text))
        intelligence['timestamp'] = datetime.now(timezone.utc).isoformat()
        intelligence['region'] = region
        intelligence['region_name'] = region_name

        return jsonify({'success': True, 'intelligence': intelligence})

    except anthropic.APIError as e:
        return jsonify({'error': f'AI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/hotspot-summary')
@require_auth
def get_hotspot_summary():
    """Generate a brief AI summary of why a specific hotspot location has an elevated or critical alert."""
    location = request.args.get('location', '').strip()
    critical = request.args.get('critical', 'false').lower() == 'true'

    if not location or len(location) > 120:
        return jsonify({'error': 'Invalid location'}), 400

    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'Anthropic API key not configured'}), 500
    if not SERPER_API_KEY and not BRAVE_API_KEY:
        return jsonify({'error': 'No search API key configured'}), 500

    risk_level = 'CRITICAL' if critical else 'ELEVATED'

    try:
        unique_results = parallel_search([
            f'{location} security conflict risk news today',
            f'{location} political instability crisis 2025',
        ], num_results=5)

        news_text = '\n'.join(
            f"- {r['title']}: {r['snippet']}"
            for r in unique_results[:10]
        )

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

        prompt = f"""You are a strategic intelligence analyst. Based on these news headlines, write a concise 2-3 sentence summary explaining the key risk factors currently driving a {risk_level} alert for {location}. Focus on the most significant security, political, economic, or humanitarian developments.

NEWS HEADLINES:
{news_text}

Respond with JSON:
{{"summary": "2-3 sentence summary of the key risks and developments warranting the {risk_level} alert."}}

Return ONLY valid JSON, no markdown or extra text."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{'role': 'user', 'content': prompt}]
        )

        data = json.loads(extract_json_object(response.content[0].text))
        return jsonify({'success': True, 'location': location, 'summary': data.get('summary', ''), 'risk_level': risk_level})

    except anthropic.APIError as e:
        return jsonify({'error': f'AI API error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/api/regional-news')
@require_auth
def get_regional_news():
    """Fetch news for a specific region."""
    region = request.args.get('region', 'world').lower().strip()

    if region not in REGION_NEWS_QUERIES:
        region = 'world'

    try:
        results = search_web(REGION_NEWS_QUERIES[region], num_results=12)
        return jsonify({'success': True, 'region': region, 'articles': results})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check."""
    return jsonify({'status': 'healthy'})


@app.route('/debug')
def debug():
    """Debug endpoint - restricted to local/dev mode only."""
    if not app.debug:
        return jsonify({'error': 'Not available in production'}), 403
    return jsonify({
        'anthropic_key_set': bool(ANTHROPIC_API_KEY),
        'serper_key_set': bool(SERPER_API_KEY),
        'brave_key_set': bool(BRAVE_API_KEY)
    })


_init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
