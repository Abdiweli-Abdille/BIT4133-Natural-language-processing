"""
KeywordIQ - AI-Driven Keyword Extraction Web Application
=========================================================
Version: 3.0 — Fixed Edition
- Auto-migrates old database (adds full_name, is_active_user columns safely)
- All admin routes working
- CSV/JSON export
"""

import os, io, json, logging, csv
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, Response, make_response)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import spacy, PyPDF2, docx

# =============================================================================
# CONFIG
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY']                  = os.environ.get('SECRET_KEY', 'keywordiq-secret-v3')
app.config['SQLALCHEMY_DATABASE_URI']     = 'sqlite:///keywordiq.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH']          = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER']              = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view         = 'login'
login_manager.login_message_category = 'info'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# spaCy
# =============================================================================
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("spaCy loaded OK.")
except OSError:
    logger.error("Run: python -m spacy download en_core_web_sm")
    nlp = None

# =============================================================================
# MODELS
# =============================================================================
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(150), unique=True, nullable=False)
    password_hash  = db.Column(db.String(256), nullable=False)
    role           = db.Column(db.String(20), nullable=False, default='user')
    full_name      = db.Column(db.String(150), nullable=True)
    is_active_user = db.Column(db.Boolean, default=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    extractions    = db.relationship('ExtractionHistory', backref='user',
                                     lazy=True, cascade='all, delete-orphan')

    def set_password(self, p):   self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)


class ExtractionHistory(db.Model):
    __tablename__ = 'extraction_history'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    source_type   = db.Column(db.String(20), nullable=False)
    filename      = db.Column(db.String(255), nullable=True)
    text_snippet  = db.Column(db.Text, nullable=False)
    keywords      = db.Column(db.Text, nullable=False)
    entity_count  = db.Column(db.Integer, default=0)
    keyword_count = db.Column(db.Integer, default=0)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(uid): return db.session.get(User, int(uid))

# =============================================================================
# DATABASE SETUP + SAFE MIGRATION
# =============================================================================
def migrate_database():
    """Add new columns to existing database without losing data."""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'keywordiq.db')
    if not os.path.exists(db_path):
        return  # New DB — SQLAlchemy will create it fresh
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        # Check existing columns
        cur.execute("PRAGMA table_info(users)")
        existing = [row[1] for row in cur.fetchall()]
        # Add missing columns safely
        if 'full_name' not in existing:
            cur.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
            logger.info("Migrated: added full_name column")
        if 'is_active_user' not in existing:
            cur.execute("ALTER TABLE users ADD COLUMN is_active_user INTEGER DEFAULT 1")
            logger.info("Migrated: added is_active_user column")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Migration warning (non-fatal): {e}")


def create_tables():
    with app.app_context():
        migrate_database()   # Upgrade old DB first
        db.create_all()      # Create any missing tables
        logger.info("Database ready.")

# =============================================================================
# HELPERS
# =============================================================================
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            if request.is_json:
                return jsonify({'error': 'Forbidden'}), 403
            flash('Access denied. Admins only.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

TARGET_ENTITIES = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}

def extract_keywords(text):
    if nlp is None: return {'entities': [], 'keywords': [], 'error': 'NLP model not loaded.'}
    if not text or not text.strip(): return {'entities': [], 'keywords': [], 'error': 'No text.'}
    doc  = nlp(text[:900_000])
    seen, ents = set(), []
    for e in doc.ents:
        if e.label_ in TARGET_ENTITIES:
            k = (e.text.strip().lower(), e.label_)
            if k not in seen:
                seen.add(k); ents.append({'text': e.text.strip(), 'label': e.label_})
    ctr = Counter()
    for t in doc:
        if t.pos_ in ('NOUN','PROPN') and not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) > 2:
            ctr[t.lemma_.lower()] += 1
    return {'entities': ents, 'keywords': [{'term':t,'count':c} for t,c in ctr.most_common(10)], 'error': None}

def allowed_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def parse_file(fs):
    fn  = secure_filename(fs.filename)
    ext = fn.rsplit('.',1)[1].lower() if '.' in fn else ''
    try:
        if ext == 'txt':
            raw = fs.read()
            try:    return raw.decode('utf-8'), None
            except: return raw.decode('latin-1'), None
        elif ext == 'pdf':
            r = PyPDF2.PdfReader(io.BytesIO(fs.read()))
            pages = [p.extract_text() for p in r.pages if p.extract_text()]
            return ('\n'.join(pages), None) if pages else ('','PDF empty.')
        elif ext == 'docx':
            d = docx.Document(io.BytesIO(fs.read()))
            paras = [p.text for p in d.paragraphs if p.text.strip()]
            return ('\n'.join(paras), None) if paras else ('','DOCX empty.')
        else: return '', f'Unsupported: .{ext}'
    except Exception as e: return '', str(e)

# =============================================================================
# PUBLIC ROUTES
# =============================================================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    text=''; fname=None; src='text'
    uf = request.files.get('document')
    pt = request.form.get('text_input','').strip()
    if uf and uf.filename:
        if not allowed_file(uf.filename):
            flash('Upload .txt, .pdf, or .docx only.','danger'); return redirect(url_for('index'))
        fname = secure_filename(uf.filename)
        text, err = parse_file(uf); src='file'
        if err: flash(f'File error: {err}','danger'); return redirect(url_for('index'))
    elif pt: text = pt
    else: flash('Paste text or upload a file.','warning'); return redirect(url_for('index'))

    res = extract_keywords(text)
    if res.get('error'): flash(res['error'],'danger'); return redirect(url_for('index'))

    if current_user.is_authenticated:
        kws = [e['text'] for e in res['entities']] + [k['term'] for k in res['keywords']]
        db.session.add(ExtractionHistory(
            user_id=current_user.id, source_type=src, filename=fname,
            text_snippet=text[:300], keywords=json.dumps(kws),
            entity_count=len(res['entities']), keyword_count=len(res['keywords'])))
        db.session.commit()

    return render_template('results.html', entities=res['entities'], keywords=res['keywords'],
                           source=src, filename=fname, char_count=len(text))

# =============================================================================
# AUTH
# =============================================================================
@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        pw    = request.form.get('password','')
        pw2   = request.form.get('confirm_password','')
        if not email or not pw:         flash('Email and password required.','danger'); return redirect(url_for('register'))
        if pw != pw2:                   flash('Passwords do not match.','danger');      return redirect(url_for('register'))
        if len(pw) < 6:                 flash('Password min 6 characters.','danger');  return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.','warning'); return redirect(url_for('register'))
        role = 'admin' if User.query.count() == 0 else 'user'
        u = User(email=email, role=role); u.set_password(pw)
        db.session.add(u); db.session.commit()
        flash(f'Account created! {"Admin access granted." if role=="admin" else "Please log in."}','success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        pw    = request.form.get('password','')
        u     = User.query.filter_by(email=email).first()
        if u and u.check_password(pw):
            login_user(u, remember=bool(request.form.get('remember')))
            flash(f'Welcome back, {u.email}!','success')
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid email or password.','danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); flash('Logged out.','info'); return redirect(url_for('index'))

# =============================================================================
# USER DASHBOARD
# =============================================================================
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin': return redirect(url_for('admin_dashboard'))
    history = ExtractionHistory.query.filter_by(user_id=current_user.id)\
                .order_by(ExtractionHistory.created_at.desc()).all()
    for r in history:
        try:    r.keyword_list = json.loads(r.keywords)
        except: r.keyword_list = []
    return render_template('dashboard.html', history=history)

# =============================================================================
# ADMIN — OVERVIEW
# =============================================================================
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    total_users       = User.query.count()
    total_extractions = ExtractionHistory.query.count()
    file_extractions  = ExtractionHistory.query.filter_by(source_type='file').count()
    text_extractions  = ExtractionHistory.query.filter_by(source_type='text').count()
    total_entities    = db.session.query(db.func.sum(ExtractionHistory.entity_count)).scalar() or 0
    global_log = ExtractionHistory.query.order_by(ExtractionHistory.created_at.desc()).limit(50).all()
    for r in global_log:
        try:    r.keyword_list = json.loads(r.keywords)
        except: r.keyword_list = []
    return render_template('admin_dashboard.html',
        stats=dict(total_users=total_users, total_extractions=total_extractions,
                   file_extractions=file_extractions, text_extractions=text_extractions,
                   total_entities=total_entities),
        global_log=global_log)

# =============================================================================
# ADMIN — ANALYTICS API
# =============================================================================
@app.route('/admin/api/analytics')
@login_required
@admin_required
def admin_analytics_api():
    today = datetime.utcnow().date()
    daily, new_users = [], []
    for i in range(13,-1,-1):
        day = today - timedelta(days=i)
        daily.append({'date': day.strftime('%b %d'),
                      'count': ExtractionHistory.query.filter(db.func.date(ExtractionHistory.created_at)==day).count()})
        new_users.append({'date': day.strftime('%b %d'),
                          'count': User.query.filter(db.func.date(User.created_at)==day).count()})
    file_c = ExtractionHistory.query.filter_by(source_type='file').count()
    text_c = ExtractionHistory.query.filter_by(source_type='text').count()
    users  = User.query.all()
    top_users = sorted([{'email': u.email, 'count': len(u.extractions)} for u in users],
                        key=lambda x: x['count'], reverse=True)[:10]
    all_rec = ExtractionHistory.query.all()
    return jsonify({
        'daily_extractions': daily, 'new_users': new_users,
        'source_split': {'file': file_c, 'text': text_c},
        'top_users': top_users,
        'totals': {'users': len(users), 'extractions': len(all_rec),
                   'entities': sum(r.entity_count for r in all_rec),
                   'keywords': sum(r.keyword_count for r in all_rec)}
    })

# =============================================================================
# ADMIN — USER CRUD
# =============================================================================
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    for u in users: u.extraction_count = ExtractionHistory.query.filter_by(user_id=u.id).count()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/add', methods=['POST'])
@login_required
@admin_required
def admin_add_user():
    data  = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    pw    = (data.get('password') or '').strip()
    role  = data.get('role','user')
    name  = (data.get('full_name') or '').strip()
    if not email or not pw:               return jsonify({'error':'Email and password required.'}), 400
    if len(pw) < 6:                       return jsonify({'error':'Password min 6 chars.'}), 400
    if User.query.filter_by(email=email).first(): return jsonify({'error':'Email already in use.'}), 409
    u = User(email=email, role=role, full_name=name or None); u.set_password(pw)
    db.session.add(u); db.session.commit()
    return jsonify({'success':True,'id':u.id,'email':u.email,'role':u.role,
                    'full_name':u.full_name or '','is_active_user':u.is_active_user,
                    'created_at':u.created_at.strftime('%Y-%m-%d'),'extraction_count':0})

@app.route('/admin/users/<int:uid>', methods=['GET'])
@login_required
@admin_required
def admin_get_user(uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error':'Not found'}), 404
    return jsonify({'id':u.id,'email':u.email,'role':u.role,'full_name':u.full_name or '',
                    'is_active_user':u.is_active_user,'created_at':u.created_at.strftime('%Y-%m-%d'),
                    'extraction_count':len(u.extractions)})

@app.route('/admin/users/<int:uid>/update', methods=['POST'])
@login_required
@admin_required
def admin_update_user(uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error':'User not found.'}), 404
    data = request.get_json() or {}
    new_email = (data.get('email') or '').strip().lower()
    new_role  = data.get('role', u.role)
    new_name  = (data.get('full_name') or '').strip()
    new_pw    = (data.get('password') or '').strip()
    new_active= data.get('is_active_user', u.is_active_user)
    if u.id == current_user.id and new_role != 'admin':
        return jsonify({'error':'Cannot remove your own admin role.'}), 400
    if new_email and new_email != u.email:
        if User.query.filter_by(email=new_email).first():
            return jsonify({'error':'Email already in use.'}), 409
        u.email = new_email
    if new_pw:
        if len(new_pw) < 6: return jsonify({'error':'Password min 6 chars.'}), 400
        u.set_password(new_pw)
    u.role = new_role; u.full_name = new_name or None; u.is_active_user = bool(new_active)
    db.session.commit()
    return jsonify({'success':True})

@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(uid):
    if uid == current_user.id: return jsonify({'error':'Cannot delete your own account.'}), 400
    u = db.session.get(User, uid)
    if not u: return jsonify({'error':'User not found.'}), 404
    db.session.delete(u); db.session.commit()
    return jsonify({'success':True})

# =============================================================================
# ADMIN — EXPORT
# =============================================================================
@app.route('/admin/export')
@login_required
@admin_required
def admin_export():
    return render_template('admin_export.html',
        total_users=User.query.count(),
        total_extractions=ExtractionHistory.query.count())

@app.route('/admin/export/users/csv')
@login_required
@admin_required
def export_users_csv():
    users = User.query.order_by(User.created_at.asc()).all()
    out   = io.StringIO(); w = csv.writer(out)
    w.writerow(['ID','Email','Full Name','Role','Active','Joined','Extractions'])
    for u in users:
        w.writerow([u.id, u.email, u.full_name or '', u.role,
                    'Yes' if u.is_active_user else 'No',
                    u.created_at.strftime('%Y-%m-%d %H:%M'), len(u.extractions)])
    out.seek(0)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment;filename=keywordiq_users_{ts}.csv'})

@app.route('/admin/export/extractions/csv')
@login_required
@admin_required
def export_extractions_csv():
    recs = ExtractionHistory.query.order_by(ExtractionHistory.created_at.desc()).all()
    out  = io.StringIO(); w = csv.writer(out)
    w.writerow(['ID','User','Source','Filename','Entities','Keywords','Keyword List','Snippet','Date'])
    for r in recs:
        try:    kw = ', '.join(json.loads(r.keywords))
        except: kw = ''
        email = r.user.email if r.user else '(deleted)'
        w.writerow([r.id, email, r.source_type, r.filename or '', r.entity_count, r.keyword_count,
                    kw, r.text_snippet[:100].replace('\n',' '), r.created_at.strftime('%Y-%m-%d %H:%M')])
    out.seek(0)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment;filename=keywordiq_extractions_{ts}.csv'})

@app.route('/admin/export/extractions/json')
@login_required
@admin_required
def export_extractions_json():
    recs = ExtractionHistory.query.order_by(ExtractionHistory.created_at.desc()).all()
    data = []
    for r in recs:
        try:    kw = json.loads(r.keywords)
        except: kw = []
        data.append({'id':r.id,'user':r.user.email if r.user else '(deleted)',
                     'source_type':r.source_type,'filename':r.filename,
                     'entity_count':r.entity_count,'keyword_count':r.keyword_count,
                     'keywords':kw,'snippet':r.text_snippet[:200],
                     'created_at':r.created_at.strftime('%Y-%m-%d %H:%M:%S')})
    ts  = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    res = make_response(json.dumps({'exported_at':datetime.utcnow().isoformat(),'total':len(data),'extractions':data}, indent=2))
    res.headers['Content-Type']        = 'application/json'
    res.headers['Content-Disposition'] = f'attachment;filename=keywordiq_extractions_{ts}.json'
    return res

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    create_tables()
    app.run(debug=True, host='0.0.0.0', port=5000)
