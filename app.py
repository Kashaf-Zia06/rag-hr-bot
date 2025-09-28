from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import csv
from datetime import datetime
from rag_hr.query import call_llm, retrieve_hr_answer
import hashlib
import re
import requests
from dotenv import load_dotenv
import requests
import json
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)
roles = ['Employee', 'Manager', 'HR', 'Director']
USERS_CSV = 'users.csv'
REQUESTS_CSV = 'requests.csv'

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL')

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user(name, email, password, role):
    exists = os.path.exists(USERS_CSV)
    with open(USERS_CSV, 'a') as f:
        if not exists:
            f.write('name,email,password,role\n')
        f.write(f'{name},{email},{hash_password(password)},{role}\n')

def get_user(email):
    if not os.path.exists(USERS_CSV):
        return None
    with open(USERS_CSV) as f:
        for line in f:
            if line.startswith('name,'): continue
            name, em, pw, role = line.strip().split(',')
            if em == email:
                return {'name': name, 'email': em, 'password': pw, 'role': role}
    return None

def get_employee_id_by_email(email):
    with open('rag_hr/data/rag_seed_data/data/employees.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['email'] == email:
                return row['employee_id']
    return None

def get_employee_by_email_and_id(email, emp_id):
    with open('rag_hr/data/rag_seed_data/data/employees.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['email'] == email and row['employee_id'] == emp_id:
                return row
    return None

def validate_signup(email, emp_id, role):
    with open('rag_hr/data/rag_seed_data/data/employees.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['email'] == email and row['employee_id'] == emp_id and row['role'] == role:
                return True
    return False

def add_employee_to_csv(employee_id, name, email, phone, role, department, location, joining_date, employment_type, manager_id):
    exists = False
    with open('rag_hr/data/rag_seed_data/data/employees.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['email'] == email or row['employee_id'] == employee_id:
                exists = True
                break
    if not exists:
        with open('rag_hr/data/rag_seed_data/data/employees.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([employee_id, name, email, phone, role, department, location, joining_date, employment_type, manager_id])
        return True
    return False

def classify_request(details):
    """
    Classify request using Groq model API.
    Returns: category string (leave, expense, attendance, overtime, promotion, transfer, resignation, travel, other)
    """
    # Check if Groq API is available
    try:
        if not GROQ_API_KEY or not GROQ_MODEL:
            return 'other'
    except NameError:
        return 'other'

    # Use the same Groq client as the chatbot
    from groq import Groq
    try:
        client = Groq(api_key="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")
        
        system_prompt = (
            "You are an HR request classifier. Your SOLE output MUST be a JSON object "
            "with a single key named 'category'. The value must be ONE of these exact words: "
            "leave, expense, attendance, overtime, promotion, transfer, resignation, travel, loan, other.\n\n"
            "Classification guidelines:\n"
            "- leave: time off, vacation, sick days, absence, holidays\n"
            "- expense: reimbursement, claims, travel costs, meals, accommodation\n"
            "- attendance: late arrival, early departure, check-in/out issues\n"
            "- overtime: extra hours, weekend work, after-hours work, overtime pay, extra hours pay\n"
            "- promotion: career advancement, salary increase, raise\n"
            "- transfer: department change, relocation, move to different role\n"
            "- resignation: quitting, leaving job, termination\n"
            "- travel: business trips, conferences, work-related travel\n"
            "- loan: financial assistance, advance salary, emergency funds\n"
            "- other: anything that doesn't fit the above categories\n\n"
            "Examples:\n"
            "- 'I want to claim my extra hours pay' → overtime\n"
            "- 'I need a loan of 10,000' → loan\n"
            "- 'I want 3 days off' → leave"
        )
        user_prompt = f"Request to classify: {details}"

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"DEBUG: Raw API response: {result}")
        
        # Parse JSON response
        import json
        try:
            parsed_data = json.loads(result)
            category = parsed_data.get('category', '').strip().lower()
            print(f"DEBUG: Parsed category: {category}")
            
            # Validate category
            valid_categories = ['leave', 'expense', 'attendance', 'overtime', 'promotion', 'transfer', 'resignation', 'travel', 'loan', 'other']
            if category in valid_categories:
                return category
            else:
                print(f"DEBUG: Invalid category '{category}', returning 'other'")
                return 'other'
                
        except json.JSONDecodeError as je:
            print(f"DEBUG: JSON decode error: {je}")
            return 'other'
            
    except Exception as e:
        # Log error but continue with fallback
        with open('groq_classification.log', 'a') as logf:
            logf.write(f"Groq classification error: {e}\n")
        print(f"DEBUG: Classification failed with error: {e}")
        print(f"DEBUG: Request details: {details}")
        return 'other'

def validate_leave_policy(req_type, details):
    # Load leave policy rules
    # For demo, only check max days and blackout dates
    max_days = {'annual': 15, 'sick': 7, 'casual': 3}
    blackout_ranges = [('06-25', '07-10'), ('11-20', '12-05')]
    # Extract days requested from details
    m = re.search(r'(\d+)\s*day', details.lower())
    days_requested = int(m.group(1)) if m else 1
    # Check max continuous days
    if req_type in max_days and days_requested > max_days[req_type]:
        return False, f"Exceeds max allowed days for {req_type} leave."
    # Check blackout dates (simplified)
    for start, end in blackout_ranges:
        if start in details or end in details:
            return False, "Requested during blackout dates."
    return True, "Valid per policy."

def get_approval_chain(req_type):
    # Load approval chain from CSV
    chain = []
    if os.path.exists('rag_hr/data/rag_seed_data/system/approval_chains.csv'):
        with open('rag_hr/data/rag_seed_data/system/approval_chains.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['request_type'] == req_type:
                    for level in ['level_1', 'level_2', 'level_3']:
                        if row[level]:
                            chain.append(row[level])
    return chain

@app.route('/', methods=['GET', 'POST'])
def select_role():
    # Redirect directly to login since role is determined from CSV
    return redirect(url_for('role_login'))

@app.route('/role_login', methods=['GET', 'POST'])
def role_login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        emp_id = request.form.get('employee_id')
        
        # Get employee data from CSV
        employee_data = get_employee_by_email_and_id(email, emp_id)
        if not employee_data:
            flash('Invalid email or employee ID. Please check your credentials and try again.', 'error')
        else:
            # Set session with data from employees.csv
            session['user'] = {
                'email': email, 
                'name': employee_data['name'],
                'role': employee_data['role']
            }
            session['role'] = employee_data['role']
            session['employee_id'] = emp_id
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)

@app.route('/role_signup', methods=['GET', 'POST'])
def role_signup():
    error = None
    if request.method == 'POST':
        # Get all form data
        employee_id = request.form.get('employee_id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', '').strip()
        department = request.form.get('department', '').strip()
        location = request.form.get('location', '').strip()
        joining_date = request.form.get('joining_date', '').strip()
        employment_type = request.form.get('employment_type', '').strip()
        manager_id = request.form.get('manager_id', '').strip()
        
        # Validate all fields are provided
        if not all([employee_id, name, email, phone, role, department, location, joining_date, employment_type, manager_id]):
            error = 'All fields are required. Please fill in all information.'
        elif get_user(email):
            error = 'Email already registered.'
        else:
            # Try to add employee to CSV
            success = add_employee_to_csv(employee_id, name, email, phone, role, department, location, joining_date, employment_type, manager_id)
            if success:
                # Save user without password
                save_user(name, email, '', role)
                return redirect(url_for('role_login'))
            else:
                error = 'Employee ID or Email already exists in the system.'
    return render_template('signup.html', error=error)

@app.route('/dashboard')
def dashboard():
    role = session.get('role')
    if not role:
        return redirect(url_for('select_role'))
    return render_template('dashboard.html', role=role)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    role = session.get('role')
    answer = None
    ref = None
    if request.method == 'POST':
        question = request.form.get('question')
        answer, citations = retrieve_hr_answer(question)
        ref = ', '.join(citations)
    return render_template('chat.html', role=role, answer=answer, ref=ref)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    role = session.get('role')
    user = session.get('user')
    employee_id = session.get('employee_id')
    attendance_records = []
    message = None
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'check_in':
            message = f'Checked in at {now}'
            with open('rag_hr/data/rag_seed_data/data/attendance_sample.csv', 'a') as f:
                f.write(f"{employee_id},{now[:10]},{now},,,0.0,Present\n")
        elif action == 'check_out':
            message = f'Checked out at {now}'
            with open('rag_hr/data/rag_seed_data/data/attendance_sample.csv', 'a') as f:
                f.write(f"{employee_id},{now[:10]},,{now},,,0.0,Present\n")
    with open('rag_hr/data/rag_seed_data/data/attendance_sample.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if employee_id and row.get('employee_id') == employee_id:
                attendance_records.append(row)
    return render_template('attendance.html', role=role, employee_id=employee_id, attendance_records=attendance_records, message=message)

@app.route('/requests', methods=['GET', 'POST'])
def requests_page():
    role = session.get('role')
    user = session.get('user')
    if not user:
        return redirect(url_for('role_login'))
    requests_list = []
    message = None
    classified_type = None
    approver = None
    if request.method == 'POST':
        details = request.form['details']
        classified_type = classify_request(details)
        status = 'Pending'
        # Leave request validation and approval chain
        if classified_type == 'leave':
            valid, msg = validate_leave_policy(classified_type, details)
            if not valid:
                message = f"Leave request invalid: {msg}"
                return render_template('requests.html', role=role, requests_list=requests_list, message=message, classified_type=classified_type, approver=approver)
        approval_chain = get_approval_chain(classified_type)
        approver = approval_chain[0] if approval_chain else 'HR'
        status = f"Pending ({' > '.join(approval_chain)})"
        with open(REQUESTS_CSV, 'a') as f:
            if os.stat(REQUESTS_CSV).st_size == 0:
                f.write('email,details,classified_type,status,approver\n')
            f.write(f"{user['email']},{details},{classified_type},{status},{approver}\n")
        message = 'Request submitted!'
    # Load requests for this user
    if os.path.exists(REQUESTS_CSV):
        with open(REQUESTS_CSV) as f:
            for line in f:
                if line.startswith('email,'): continue
                parts = line.strip().split(',', 4)
                if len(parts) == 5:
                    email, details, classified_type, status, approver = parts
                else:
                    email, details, classified_type, status = parts
                    approver = 'HR'
                if user and email == user['email']:
                    requests_list.append({'details': details, 'classified_type': classified_type, 'status': status, 'approver': approver})
    
    # Only show success box if a request was just submitted
    show_success = request.method == 'POST' and classified_type is not None
    return render_template('requests.html', role=role, requests_list=requests_list, message=message, classified_type=classified_type if show_success else None, approver=approver if show_success else None)

@app.route('/hr_requests')
def hr_requests():
    role = session.get('role')
    if role != 'HR':
        return redirect(url_for('dashboard'))
    all_requests = []
    if os.path.exists(REQUESTS_CSV):
        with open(REQUESTS_CSV) as f:
            for line in f:
                if line.startswith('email,'): continue
                parts = line.strip().split(',', 4)
                if len(parts) == 5:
                    email, details, classified_type, status, approver = parts
                else:
                    email, details, classified_type, status = parts
                    approver = 'HR'
                all_requests.append({'email': email, 'details': details, 'classified_type': classified_type, 'status': status, 'approver': approver})
    return render_template('hr_requests.html', role=role, all_requests=all_requests)

@app.route('/approvals', methods=['GET', 'POST'])
def approvals():
    role = session.get('role')
    if role != 'HR':
        return redirect(url_for('dashboard'))
    
    message = None
    error = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        request_id = request.form.get('request_id')
        
        if action and request_id:
            # Update request status in CSV
            if os.path.exists(REQUESTS_CSV):
                # Read all requests
                requests_data = []
                with open(REQUESTS_CSV, 'r') as f:
                    lines = f.readlines()
                
                # Process each line
                for i, line in enumerate(lines):
                    if line.startswith('email,'):
                        requests_data.append(line)
                        continue
                    
                    parts = line.strip().split(',', 4)
                    if len(parts) >= 4:
                        email, details, classified_type, status, approver = parts[0], parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else 'HR'
                        
                        # Check if this is the request to update
                        if str(i) == request_id:
                            if action == 'approve':
                                new_status = 'Approved'
                                message = f"Request approved successfully!"
                            elif action == 'disapprove':
                                new_status = 'Disapproved'
                                message = f"Request disapproved."
                            else:
                                error = "Invalid action"
                                break
                            
                            requests_data.append(f"{email},{details},{classified_type},{new_status},{approver}\n")
                        else:
                            requests_data.append(line)
                
                # Write back to CSV
                with open(REQUESTS_CSV, 'w') as f:
                    f.writelines(requests_data)
            else:
                error = "No requests file found"
    
    # Load all requests for HR
    all_requests = []
    if os.path.exists(REQUESTS_CSV):
        with open(REQUESTS_CSV, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('email,'):
                    continue
                
                parts = line.strip().split(',', 4)
                if len(parts) >= 4:
                    email, details, classified_type, status, approver = parts[0], parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else 'HR'
                    
                    # Show all requests (pending, approved, disapproved)
                    all_requests.append({
                        'id': i,
                        'email': email,
                        'details': details,
                        'classified_type': classified_type,
                        'status': status,
                        'approver': approver,
                        'timestamp': 'Recently'  # You can add actual timestamp if needed
                    })
    
    return render_template('approvals.html', role=role, all_requests=all_requests, message=message, error=error)

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    emp_id_search = request.args.get('search_id') or request.form.get('search_id')
    # Requests by Type and Status
    req_types = {}
    req_status = {'Approved': 0, 'Pending': 0, 'Cancelled': 0}
    req_type_status = {}
    employee_requests = []
    grouped_by_emp = {}
    if os.path.exists('rag_hr/data/rag_seed_data/data/requests_sample.csv'):
        with open('rag_hr/data/rag_seed_data/data/requests_sample.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emp_id = row.get('employee_id', 'Unknown')
                t = row.get('request_type', 'Unknown')
                s = row.get('status', 'Unknown')
                req_types[t] = req_types.get(t, 0) + 1
                if s in req_status:
                    req_status[s] += 1
                if t not in req_type_status:
                    req_type_status[t] = {'Approved': 0, 'Pending': 0, 'Cancelled': 0}
                if s in req_type_status[t]:
                    req_type_status[t][s] += 1
                if emp_id not in grouped_by_emp:
                    grouped_by_emp[emp_id] = {'Approved': 0, 'Pending': 0, 'Cancelled': 0}
                if s in grouped_by_emp[emp_id]:
                    grouped_by_emp[emp_id][s] += 1
                if emp_id_search and emp_id == emp_id_search:
                    employee_requests.append(row)
    request_types = list(req_types.keys())
    request_counts = list(req_types.values())
    status_labels = list(req_status.keys())
    status_counts = list(req_status.values())
    grouped_status = {k: [v['Approved'], v['Pending'], v['Cancelled']] for k, v in req_type_status.items()}
    grouped_emp_status = {k: [v['Approved'], v['Pending'], v['Cancelled']] for k, v in grouped_by_emp.items()}
    # Performance Metrics
    perf_data = []
    if os.path.exists('rag_hr/data/rag_seed_data/data/performance_metrics_sample.csv'):
        with open('rag_hr/data/rag_seed_data/data/performance_metrics_sample.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                perf_data.append({
                    'employee_id': row.get('employee_id', 'Unknown'),
                    'task_completion_rate': float(row.get('task_completion_rate', 0)),
                    'hours_logged': float(row.get('hours_logged', 0)),
                    'contribution_score': float(row.get('contribution_score', 0))
                })
    filtered_perf = None
    if emp_id_search:
        filtered = [emp for emp in perf_data if emp['employee_id'] == emp_id_search]
        filtered_perf = filtered[0] if filtered else None
    # Attendance Data
    attendance_summary = {}
    employee_attendance = []
    if os.path.exists('rag_hr/data/rag_seed_data/data/attendance_sample.csv'):
        with open('rag_hr/data/rag_seed_data/data/attendance_sample.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emp_id = row.get('employee_id', 'Unknown')
                if row.get('status') == 'Present':
                    attendance_summary[emp_id] = attendance_summary.get(emp_id, 0) + 1
                if emp_id_search and emp_id == emp_id_search:
                    employee_attendance.append(row)
    return render_template('analytics.html',
        request_types=request_types,
        request_counts=request_counts,
        status_labels=status_labels,
        status_counts=status_counts,
        grouped_status=grouped_status,
        grouped_emp_status=grouped_emp_status,
        perf_data=perf_data,
        search_id=emp_id_search,
        searched_employee=filtered_perf,
        employee_requests=employee_requests,
        attendance_summary=attendance_summary,
        employee_attendance=employee_attendance)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        emp_id = request.form.get('employee_id')
        if get_user(email):
            error = 'Email already registered.'
        elif not validate_signup(email, emp_id, role):
            add_employee_to_csv(name, email, emp_id, role)
        save_user(name, email, password, role)
        return redirect(url_for('login'))
    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        emp_id = request.form.get('employee_id')
        user = get_user(email)
        valid_employee = True
        if user and user['role'] == 'Employee':
            valid_employee = bool(get_employee_by_email_and_id(email, emp_id))
        if not user or user['password'] != hash_password(password) or not valid_employee:
            error = 'Invalid email, password, or employee ID.'
        else:
            session['user'] = user
            session['role'] = user['role']
            session['employee_id'] = emp_id if emp_id else get_employee_id_by_email(email)
            return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/hr_analytics')
def hr_analytics():
    # Requests by Type
    req_types = {}
    if os.path.exists('rag_hr/data/rag_seed_data/data/requests_sample.csv'):
        with open('rag_hr/data/rag_seed_data/data/requests_sample.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get('request_type', 'Unknown')
                req_types[t] = req_types.get(t, 0) + 1
    request_types = list(req_types.keys())
    request_counts = list(req_types.values())
    # Performance Metrics
    perf_names = []
    perf_scores = []
    if os.path.exists('rag_hr/data/rag_seed_data/data/performance_metrics_sample.csv'):
        with open('rag_hr/data/rag_seed_data/data/performance_metrics_sample.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                perf_names.append(row.get('employee_id', 'Unknown'))
                perf_scores.append(float(row.get('score', 0)))
    return render_template('hr_analytics.html',
        request_types=request_types,
        request_counts=request_counts,
        perf_names=perf_names,
        perf_scores=perf_scores)

if __name__ == '__main__':
    app.run(debug=True)
