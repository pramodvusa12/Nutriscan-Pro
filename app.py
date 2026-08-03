from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from flask_mail import Mail, Message
import random
# ---- AI imports (model inference) ----
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# =============================================================================
# Flask + DB setup
# =============================================================================
app = Flask(__name__)
# =======================
# Gmail Configuration
# =======================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'dharshiniyendluri13@gmail.com'
app.config['MAIL_PASSWORD'] = 'hbhehofccapysymj'
app.config['MAIL_DEFAULT_SENDER'] = 'dharshiniyendluri13@gmail.com'

mail = Mail(app)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config['UPLOAD_FOLDER']                  = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']             = 10 * 1024 * 1024
app.config['SECRET_KEY']                     = 'nutriscan_secret'
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# =============================================================================
# Database models
# =============================================================================
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email    = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class Scan(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(100))
    image_path = db.Column(db.String(300))
    result     = db.Column(db.String(100))
    confidence = db.Column(db.Float, default=0.0)
    date       = db.Column(db.String(100))

# =============================================================================
# PyTorch model — loaded once at startup
# =============================================================================
MODEL_PATH   = 'fresh_spoiled_food_model.pt'
INFER_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FoodClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = 'tf_efficientnetv2_b0'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            num_classes=0, global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


_model       = None
_class_names = None
_eval_tf     = None


def load_model():
    global _model, _class_names, _eval_tf

    if not os.path.exists(MODEL_PATH):
        print(f'⚠️  {MODEL_PATH} not found. AI analysis will be disabled.')
        return

    try:
        ckpt = torch.load(MODEL_PATH, map_location=INFER_DEVICE)
        _class_names = ckpt['class_names']

        _model = FoodClassifier(
            num_classes=len(_class_names),
            backbone_name=ckpt.get('backbone_name', 'tf_efficientnetv2_b0'),
        )
        _model.load_state_dict(ckpt['model_state_dict'])
        _model.eval().to(INFER_DEVICE)

        img_size = ckpt.get('img_size', 224)
        mean     = ckpt.get('normalize_mean', [0.485, 0.456, 0.406])
        std      = ckpt.get('normalize_std',  [0.229, 0.224, 0.225])
        _eval_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        print(f'✅ Model loaded ({len(_class_names)} classes) on {INFER_DEVICE}')
        print(f'   Classes: {_class_names}')

    except Exception as e:
        print(f'❌ Failed to load model: {e}')
        _model = None


@torch.no_grad()
def predict_image(filepath: str, tta: bool = True):
    if _model is None:
        return None, None

    img = Image.open(filepath).convert('RGB')
    x   = _eval_tf(img).unsqueeze(0).to(INFER_DEVICE)

    p = F.softmax(_model(x), dim=1)
    if tta:
        p_flip = F.softmax(_model(torch.flip(x, dims=[3])), dim=1)
        p = (p + p_flip) / 2

    p   = p.cpu().numpy()[0]
    idx = int(p.argmax())
    return _class_names[idx], float(p[idx])

# =============================================================================
# Helpers
# =============================================================================
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


_FRESH_PREFIXES   = ('fresh', 'good')
_SPOILED_PREFIXES = ('rotten', 'spoiled', 'stale', 'bad', 'mold')

def parse_class(label: str):
    """Turn a class name like 'freshapples' / 'rotten_banana' into
    (status, food_key). Best-effort — falls back to ('unknown', label)."""
    norm = label.lower().replace('_', '').replace(' ', '').replace('-', '')

    for p in _FRESH_PREFIXES:
        if norm.startswith(p):
            return 'fresh', _singularize(norm[len(p):])
    for p in _SPOILED_PREFIXES:
        if norm.startswith(p):
            return 'spoiled', _singularize(norm[len(p):])

    return 'unknown', _singularize(norm)

PRESERVATION_DAYS = {
    "bread": 3,
    "dairy": 5,
    "fruit": 7,
    "vegetable": 5
}

def _singularize(s: str) -> str:
    if s.endswith('ies'): return s[:-3] + 'y'
    if s.endswith('oes'): return s[:-2]
    if s.endswith('s') and not s.endswith('ss'): return s[:-1]
    return s


def pretty_label(label: str) -> str:
    status, food = parse_class(label)
    if status == 'unknown':
        return label.replace('_', ' ').title()
    return f"{status.title()} {food.title()}"


def ensure_schema():
    print("Creating database tables...")

    db.create_all()

    print("Database tables created.")

    try:
        with db.engine.connect() as conn:
            conn.execute(db.text(
                'ALTER TABLE scan ADD COLUMN confidence FLOAT DEFAULT 0.0'
            ))
            conn.commit()
    except Exception as e:
        print("ALTER TABLE skipped:", e)
# =============================================================================
# Routes — pages
# =============================================================================
@app.route('/')
def home():
    return render_template('index.html', username=session.get('user'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip().lower()
        password = request.form['password']

        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash('Username or email already in use.', 'error')
            return render_template('signup.html')

        user = User(username=username, email=email,
                    password=generate_password_hash(password))
        db.session.add(user); db.session.commit()

        flash('Account created. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            return redirect(url_for('dashboard'))

        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("Email not found.", "error")
            return redirect(url_for('forgot_password'))

        otp = str(random.randint(100000, 999999))

        otp_storage[email] = otp
        session['reset_email'] = email

        msg = Message(
            "NutriScan Pro Password Reset",
            recipients=[email]
        )

        msg.body = f"""
Hello,

Your NutriScan Pro password reset OTP is:

{otp}

This OTP is valid for one use.

Thank you,
NutriScan Pro
"""

        print("Receiver email:", user.email)

        try:
            mail.send(msg)
            print("MAIL SENT SUCCESSFULLY")
        except Exception as e:
            print("MAIL ERROR:", e)

        flash("OTP has been sent to your email.", "success")
        return redirect(url_for('verify_otp'))

    return render_template('forgot_password.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')

    recent_scans = (Scan.query.filter_by(username=session['user'])
                              .order_by(Scan.id.desc()).limit(5).all())
    total_scans  = Scan.query.filter_by(username=session['user']).count()

    return render_template(
        'dashboard.html',
        username=session['user'],
        recent_scans=recent_scans,
        total_scans=total_scans,
        model_loaded=(_model is not None),
    )


@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')
    user = User.query.filter_by(username=session['user']).first()
    scan_count = Scan.query.filter_by(username=session['user']).count()
    return render_template('profile.html', username=session['user'],
                           user=user, scan_count=scan_count)


@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')
    scans = (Scan.query.filter_by(username=session['user'])
                       .order_by(Scan.id.desc()).all())
    return render_template('history.html', username=session['user'], scans=scans)
@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():

    if 'reset_email' not in session:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        entered_otp = request.form['otp']
        email = session['reset_email']

        if otp_storage.get(email) == entered_otp:
            return redirect(url_for('reset_password'))
        else:
            flash("Invalid OTP", "error")

    return render_template('verify_otp.html')
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():

    if 'reset_email' not in session:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form['password']
        email = session['reset_email']

        user = User.query.filter_by(email=email).first()

        if user:
            user.password = generate_password_hash(new_password)
            db.session.commit()

            otp_storage.pop(email, None)
            session.pop('reset_email', None)

            flash("Password reset successful. Please login.", "success")
            return redirect(url_for('login'))

    return render_template('reset_password.html')
# =============================================================================
# Upload — JSON API (for AJAX from index.html)
# =============================================================================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'user' not in session:
        return jsonify({'error': 'not authenticated', 'login_url': '/login'}), 401

    image = request.files.get('image')
    if not image or image.filename == '':
        return jsonify({'error': 'No image provided'}), 400
    if not allowed_file(image.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    safe_name   = secure_filename(image.filename)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_name = f"{session['user']}_{timestamp}_{safe_name}"
    filepath    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    image.save(filepath)

    label, confidence = predict_image(filepath, tta=True)
    if label is None:
        return jsonify({'error': 'Model not loaded on server'}), 503

    status, food = parse_class(label)
    display = pretty_label(label)

    if status == "spoiled":
        preservation_days = "Expired"
    else:
        preservation_days = PRESERVATION_DAYS.get(food, "Unknown")

    scan = Scan(
        username   = session['user'],
        image_path = filepath.replace(os.sep, '/'),
        result     = display,
        confidence = confidence,
        date       = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    db.session.add(scan); db.session.commit()

    return jsonify({
        'label':      label,
        'display':    display,
        'food':       food,
        'status':     status,           # 'fresh' | 'spoiled' | 'unknown'
        'confidence': round(confidence, 4),
        'preservation_days': preservation_days,
        'image_url':  '/' + filepath.replace(os.sep, '/'),
    })

# =============================================================================
# Upload — old form fallback (still works if JS is disabled)
# =============================================================================
@app.route('/upload', methods=['POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')

    image = request.files.get('image')
    if not image or image.filename == '':
        flash('No image selected.', 'error')
        return redirect(url_for('dashboard'))
    if not allowed_file(image.filename):
        flash('Unsupported file type.', 'error')
        return redirect(url_for('dashboard'))

    safe_name   = secure_filename(image.filename)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_name = f"{session['user']}_{timestamp}_{safe_name}"
    filepath    = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    image.save(filepath)

    label, confidence = predict_image(filepath, tta=True)
    if label is None:
        result_text, confidence = 'Model not loaded — image saved only', 0.0
        flash(result_text, 'warning')
    else:
        result_text = pretty_label(label)
        flash(f'Result: {result_text}  ({confidence*100:.1f}% confidence)', 'success')

    scan = Scan(
        username   = session['user'],
        image_path = filepath.replace(os.sep, '/'),
        result     = result_text,
        confidence = confidence,
        date       = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    db.session.add(scan); db.session.commit()

    return redirect(url_for('dashboard'))

# =============================================================================
if __name__ == '__main__':
    app.run(debug=True)