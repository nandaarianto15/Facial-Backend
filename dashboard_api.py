import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import re

# IMPORT DARI FILE DATABASE TERPISAH
from database import Base, engine, SessionLocal, get_db, User, Blog

# ==========================================
# KONFIGURASI PATH
# ==========================================
# Folder utama untuk menyimpan gambar blog
BLOG_IMAGE_DIR = "blog_image"
# Pastikan folder utama ada
os.makedirs(BLOG_IMAGE_DIR, exist_ok=True)

# ==========================================
# PYDANTIC MODELS (Schema)
# ==========================================

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    tel: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Schema untuk input teks saja (tidak dipakai di create jika pakai UploadFile, tapi berguna untuk update parsial)
class ArticleBase(BaseModel):
    title: str
    category: str
    tag: str
    image: str
    content: str
    is_published: bool = True

class ArticleCreate(BaseModel):
    author_id: int = 1 

class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    tag: Optional[str] = None
    image: Optional[str] = None
    content: Optional[str] = None
    is_published: Optional[bool] = None

class ArticleResponse(BaseModel):
    id: int
    title: str
    slug: str
    category: str
    tag: str
    image: str
    content: str
    is_published: bool
    published_at: datetime
    author_id: int
    created_at: datetime # Tambahkan ini
    
    class Config:
        from_attributes = True

# ==========================================
# ROUTER INIT
# ==========================================
router = APIRouter()

# ==========================================
# ENDPOINT CRM
# ==========================================

@router.get("/crm/users", response_model=List[UserResponse])
def get_all_users(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    users = db.query(User).order_by(desc(User.created_at)).offset(skip).limit(limit).all()
    return users

@router.get("/crm/users/{user_id}", response_model=UserResponse)
def get_user_detail(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ==========================================
# ENDPOINT BLOG
# ==========================================

def generate_slug(title: str):
    slug = re.sub(r'[^\w\s-]', '', title.lower()).strip().replace(' ', '-')
    return re.sub(r'-+', '-', slug)

# CREATE ARTICLE - MENGgunakan Form Data & UploadFile
@router.post("/blog/create", response_model=ArticleResponse)
async def create_article(
    title: str = Form(...),
    category: str = Form(...),
    tag: str = Form(...),
    content: str = Form(...),
    is_published: bool = Form(True),
    author_id: int = Form(1),
    image: UploadFile = File(...), # Wajib kirim file
    db: Session = Depends(get_db)
):
    # 1. Generate Slug
    base_slug = generate_slug(title)
    slug = base_slug
    counter = 1
    while db.query(Blog).filter(Blog.slug == slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1

    # 2. Set Timestamps
    now = datetime.now()
    
    # 3. Buat Direktori Khusus untuk Artikel Ini
    # Format: blog_image/{slug}-{timestamp}
    # Timestamp format: YYYYMMDDHHMMSS
    time_str = now.strftime("%Y%m%d%H%M%S")
    folder_name = f"{slug}-{time_str}"
    article_dir = os.path.join(BLOG_IMAGE_DIR, folder_name)
    
    try:
        os.makedirs(article_dir, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Gagal membuat direktori: {str(e)}")

    # 4. Simpan Gambar
    # Ambil ekstensi file
    file_ext = os.path.splitext(image.filename)[1] or ".jpg"
    # Nama file standar (misal: image.jpg)
    file_name = f"image{file_ext}"
    file_path = os.path.join(article_dir, file_name)
    
    # Path yang disimpan di DB (relatif dari root project atau URL statis)
    # Misal: /static/blog_image/slug-time/image.jpg
    # Atau path relatif sistem: blog_image/slug-time/image.jpg
    # Di sini kita simpan path relatif yang bisa diakses web nanti
    db_image_path = f"{article_dir}/{file_name}" 

    # Tulis file ke disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan gambar: {str(e)}")

    # 5. Simpan ke Database
    new_article = Blog(
        title=title,
        slug=slug,
        content=content,
        category=category,
        tag=tag,
        image=db_image_path,
        is_published=is_published,
        author_id=author_id,
        published_at=now if is_published else None,
        created_at=now # Set created_at manual agar sinkron dengan folder
    )
    
    db.add(new_article)
    db.commit()
    db.refresh(new_article)
    return new_article

@router.get("/blog/list", response_model=List[ArticleResponse])
def get_all_articles(
    is_published: Optional[bool] = None,
    category: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    query = db.query(Blog)
    
    if is_published is not None:
        query = query.filter(Blog.is_published == is_published)
    
    if category:
        query = query.filter(Blog.category == category)

    return query.order_by(desc(Blog.published_at)).limit(limit).all()

@router.get("/blog/{slug}", response_model=ArticleResponse)
def get_article_by_slug(slug: str, db: Session = Depends(get_db)):
    article = db.query(Blog).filter(Blog.slug == slug).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

# UPDATE ARTICLE - Bisa update teks dan/atau gambar
@router.put("/blog/{article_id}", response_model=ArticleResponse)
async def update_article(
    article_id: int, 
    title: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tag: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    is_published: Optional[bool] = Form(None),
    image: Optional[UploadFile] = File(None), # Opsional, kalau mau ganti gambar
    db: Session = Depends(get_db)
):
    article = db.query(Blog).filter(Blog.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    # Update teks jika ada
    if title is not None:
        article.title = title
        # Optional: Update slug jika title berubah? Sebaiknya jangan untuk menjaga SEO, kecuali diminta.
    
    if content is not None: article.content = content
    if category is not None: article.category = category
    if tag is not None: article.tag = tag
    
    if is_published is not None:
        article.is_published = is_published
        if is_published and not article.published_at:
             article.published_at = datetime.now()

    # Update Gambar jika ada file baru
    if image:
        # Hapus gambar lama jika ada (opsional, hemat space)
        if article.image and os.path.exists(article.image):
            try:
                # Hapus file atau bahkan folder lama?
                # Disini kita hanya hapus filenya, atau bisa replace di folder yang sama.
                # Untuk simple-nya, kita replace file di folder yang sama.
                pass 
            except:
                pass
        
        # Jika artikel belum punya folder (kasus jarang), buat baru
        if not article.image or not os.path.exists(os.path.dirname(article.image)):
            now = datetime.now()
            time_str = now.strftime("%Y%m%d%H%M%S")
            base_slug = generate_slug(article.title)
            folder_name = f"{base_slug}-{time_str}"
            article_dir = os.path.join(BLOG_IMAGE_DIR, folder_name)
            os.makedirs(article_dir, exist_ok=True)
            db_image_path = os.path.join(article_dir, f"image{os.path.splitext(image.filename)[1]}")
        else:
            # Gunakan path lama
            db_image_path = article.image
            # Hapus file lama dulu
            if os.path.exists(db_image_path):
                os.remove(db_image_path)
            # Update ekstensi jika berubah
            db_image_path = os.path.join(os.path.dirname(db_image_path), f"image{os.path.splitext(image.filename)[1]}")

        # Simpan file baru
        with open(db_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        article.image = db_image_path

    db.commit()
    db.refresh(article)
    return article

@router.delete("/blog/{article_id}")
def delete_article(article_id: int, db: Session = Depends(get_db)):
    article = db.query(Blog).filter(Blog.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Hapus folder gambar jika ada
    if article.image:
        folder_path = os.path.dirname(article.image)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path) # Hapus 1 folder artikel tersebut
            except Exception as e:
                print(f"Error deleting folder: {e}")

    db.delete(article)
    db.commit()
    return {"success": True, "message": "Article deleted successfully"}