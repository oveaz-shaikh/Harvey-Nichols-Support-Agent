from pathlib import Path
import os, re, json, base64
from typing import Optional, List
import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
load_dotenv(BACKEND_DIR / '.env')
MASTER_PATH = Path(os.getenv('MASTER_CATALOGUE_PATH', 'data/products_master.csv'))
if not MASTER_PATH.is_absolute(): MASTER_PATH = BACKEND_DIR / MASTER_PATH
RAW_PATH = Path(os.getenv('CATALOGUE_PATH', 'data/products.csv'))
if not RAW_PATH.is_absolute(): RAW_PATH = BACKEND_DIR / RAW_PATH
CLEAN_PATH = Path(os.getenv('CLEAN_CATALOGUE_PATH', 'data/products_clean.csv'))
if not CLEAN_PATH.is_absolute(): CLEAN_PATH = BACKEND_DIR / CLEAN_PATH
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
app = FastAPI(title='Harvey Nichols AI Luxury Shopping Assistant', version='2.0.0', description='Metadata-enriched, catalogue-grounded luxury shopping prototype. Hard constraints are applied before AI explanations.')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
_df = None

class OutfitRequest(BaseModel):
    occasion: str = Field(default='casual weekend', max_length=100)
    style: str = Field(default='smart casual', max_length=100)
    budget: float = Field(gt=0, le=100000)
    colour: Optional[str] = Field(default=None, max_length=50)
    audience: str = Field(default='any', max_length=30)
    season: str = Field(default='any', max_length=30)
    include_accessories: bool = True
class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    budget: Optional[float] = Field(default=None, gt=0, le=100000)
class GiftRequest(BaseModel):
    recipient: str = Field(default='someone special', max_length=100)
    recipient_audience: str = Field(default='any', max_length=20)
    occasion: str = Field(default='thoughtful gift', max_length=100)
    sentiment: str = Field(default='appreciation', max_length=200)
    interests: str = Field(default='', max_length=300)
    preferred_category: str = Field(default='any', max_length=50)
    style_preference: str = Field(default='', max_length=200)
    avoid: str = Field(default='', max_length=300)
    budget: float = Field(gt=0, le=100000)
class ProductQuestion(BaseModel):
    product_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=800)
class WardrobeRequest(BaseModel):
    product_ids: List[str] = Field(default_factory=list, max_length=100)


def _truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return norm(value) in {"true", "1", "yes", "y"}


def catalogue():
    """Load the curated master catalogue first, then legacy enriched/raw files.

    The adapter maps the new lower_snake_case CSV schema to legacy field names so
    existing API responses remain compatible while the frontend is migrated.
    """
    global _df
    if _df is None:
        if MASTER_PATH.exists():
            path = MASTER_PATH
            schema = "master"
        elif CLEAN_PATH.exists():
            path = CLEAN_PATH
            schema = "legacy"
        else:
            path = RAW_PATH
            schema = "legacy"

        if not path.exists():
            raise HTTPException(500, f"Catalogue file not found: {path}")

        try:
            df = pd.read_csv(path, low_memory=False)

            if schema == "master":
                aliases = {
                    "product_id": "Product_ID",
                    "product_name": "Product_Name",
                    "brand": "Brand",
                    "category": "Category",
                    "subcategory": "Subcategory",
                    "product_type": "Product_Type_Normalized",
                    "audience": "Audience",
                    "colour": "Colour",
                    "material": "Material",
                    "fit": "Fit",
                    "style_tags": "Style_Tags",
                    "occasion_tags": "Occasion_Tags",
                    "season_tags": "Season_Tags",
                    "outfit_role": "Outfit_Role",
                    "selling_price_gbp": "Selling_Price_GBP",
                    "ticket_price_gbp": "Ticket_Price_GBP",
                    "discount_percentage": "Discount_Percentage",
                    "available_sizes": "Available_Sizes",
                    "image_url": "Image_URL",
                    "product_description": "Product_Description",
                    "care_instructions": "Care_Instructions",
                    "country_of_manufacture": "Country_Of_Manufacture",
                    "giftable": "Giftable",
                    "gift_category": "Gift_Category",
                    "source_product_id": "Source_Product_ID",
                    "data_quality_status": "Data_Quality_Status",
                    "metadata_source": "Metadata_Source",
                }
                for source_col, legacy_col in aliases.items():
                    if source_col in df and legacy_col not in df:
                        df[legacy_col] = df[source_col]

                # Exclude records that were not approved for recommendations.
                if "Data_Quality_Status" in df:
                    status = df["Data_Quality_Status"].fillna("review").astype(str).str.lower()
                    # Keep review records available for broad discovery, but preserve
                    # the review flag so stricter recommendation rules can avoid them.
                    df = df[~status.eq("excluded")].copy()
                    df["Needs_Review"] = status.loc[df.index].eq("review")
                else:
                    df["Needs_Review"] = False

                if "Giftable" in df:
                    df["Giftable"] = df["Giftable"].map(_truthy)

                if "Colour" in df and "Normalized_Colour" not in df:
                    df["Normalized_Colour"] = df["Colour"].fillna("").astype(str).str.lower().str.strip()

                # Master schema has explicit roles; normalize unsupported roles to None.
                if "Outfit_Role" in df:
                    df["Outfit_Role"] = df["Outfit_Role"].fillna("unknown").astype(str).str.lower().str.strip()
            else:
                # Normalize legacy boolean-like fields from CSV.
                if "Giftable" in df:
                    df["Giftable"] = df["Giftable"].map(_truthy)

            for c in ["Selling_Price_GBP", "Ticket_Price_GBP", "Discount_Percentage", "Discount_Value"]:
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            if "Product_ID" not in df or "Product_Name" not in df:
                raise ValueError("Catalogue must contain product ID and product name fields")
            if "Selling_Price_GBP" not in df:
                raise ValueError("Catalogue must contain selling_price_gbp / Selling_Price_GBP")

            df["Product_ID"] = df["Product_ID"].astype(str).str.strip()
            df = df[df["Product_ID"].ne("") & df["Product_Name"].notna()].copy()
            df = df[df["Selling_Price_GBP"].notna() & (df["Selling_Price_GBP"] > 0)].copy()
            df = df.drop_duplicates("Product_ID", keep="first")

            if "Outfit_Role" in df:
                excluded_roles = {
                    "excluded", "unknown", "excluded_from_outfits", "unclassified",
                    "clothing_review", "accessory_review", "beauty", "gift_item"
                }
                role_values = df["Outfit_Role"].fillna("unknown").astype(str).str.lower().str.strip()
                df["_role_key"] = role_values.where(~role_values.isin(excluded_roles), None)
            else:
                df["_role_key"] = df.apply(infer_role, axis=1)

            df["_audience_key"] = (
                df["Audience"].fillna("unknown").astype(str).str.lower().str.strip()
                if "Audience" in df else df.apply(audience_of, axis=1)
            )
            df["_price_key"] = pd.to_numeric(df["Selling_Price_GBP"], errors="coerce")
            df["_catalogue_source"] = path.name
            _df = df.reset_index(drop=True)
        except Exception as e:
            raise HTTPException(500, f"Could not load catalogue {path}: {e}")
    return _df

def val(v):
    if pd.isna(v): return None
    if hasattr(v, 'item'): v = v.item()
    if isinstance(v, (int, float, bool)): return v
    return str(v).strip()
def product_dict(row):
    d = {k: val(v) for k, v in row.items() if not str(k).startswith('_')}
    d['Image_URL'] = d.get('Image_URL') or d.get('Best_Image_URL') or d.get('Main_Image_URL')
    if not d.get('Image_URL') and d.get('All_Image_URLs'):
        d['Image_URL'] = str(d['All_Image_URLs']).split('|')[0].strip() or None
    d['Product_URL'] = d.get('Product_URL') or d.get('Product_URL_GBP')
    # Keep the normalized schema visible to newer clients while retaining legacy keys.
    normalized_aliases = {
        'product_id': 'Product_ID', 'product_name': 'Product_Name',
        'brand': 'Brand', 'category': 'Category', 'subcategory': 'Subcategory',
        'product_type': 'Product_Type_Normalized', 'audience': 'Audience',
        'colour': 'Colour', 'outfit_role': 'Outfit_Role',
        'selling_price_gbp': 'Selling_Price_GBP', 'image_url': 'Image_URL',
        'giftable': 'Giftable', 'gift_category': 'Gift_Category',
    }
    for normalized, legacy in normalized_aliases.items():
        if normalized not in d and legacy in d:
            d[normalized] = d.get(legacy)
        if legacy not in d and normalized in d:
            d[legacy] = d.get(normalized)
    return d
def safe_products(df, limit=20): return [product_dict(r) for _, r in df.head(limit).iterrows()]
def price_series(df): return pd.to_numeric(df.get('Selling_Price_GBP', pd.Series(index=df.index, dtype=float)), errors='coerce')
def norm(v): return re.sub(r'\s+', ' ', str(v or '').lower().replace('’', "'").replace('–','-')).strip()
def text_for(row): return ' '.join(str(row.get(c) or '') for c in ['Product_Name','Brand','Web_Brand','Category','Subcategory','Colour','Activity','Activity_Group','Product_Type_Normalized','Outfit_Role','Material','Fit','Style_Tags','Occasion_Tags','Season_Tags','Product_Description','Gift_Category']).lower()

STOP = {'find','show','give','want','need','looking','look','for','with','under','over','the','and','some','please','me','my','a','an','of','on','in','to','that','what','have','mind','something','nice','good','best','get','buy','shop','product','products','item','items','catalogue','catalog'}
ROLE_PATTERNS = {
 'bag': r'\b(bag|bags|handbag|handbags|tote|totes|backpack|backpacks|crossbody|clutch|shoulder bag|bum bag)\b',
 'footwear': r'\b(shoe|shoes|trainer|trainers|sneaker|sneakers|boots|boot|loafers|sandal|sandals|heels|footwear|runners)\b',
 'upperwear': r'\b(shirt|shirts|t-shirt|tee|top|tops|blouse|jumper|sweater|hoodie|hoodies|jacket|coat|blazer|polo|vest)\b',
 'lowerwear': r'\b(trousers|pants|jeans|jean|shorts|skirt|skirts|joggers|leggings|chinos)\b',
 'one_piece': r'\b(dress|dresses|gown|jumpsuit|playsuit)\b',
 'accessory': r'\b(watch|watches|belt|belts|cap|caps|hat|hats|scarf|scarves|sunglasses|jewellery|jewelry|necklace|bracelet|earrings|wallet|wallets|tie|ties)\b',
 'beauty_home': r'\b(perfume|fragrance|parfum|candle|skincare|makeup|lip balm|body lotion)\b'
}
def infer_role(row):
    if 'Outfit_Role' in row and pd.notna(row.get('Outfit_Role')):
        role = norm(row.get('Outfit_Role'))
        excluded = {'excluded', 'unknown', 'excluded_from_outfits', 'unclassified',
                    'clothing_review', 'accessory_review', 'beauty', 'gift_item'}
        return role if role not in excluded else None
    t = norm(f"{row.get('Product_Name','')} {row.get('Subcategory','')} {row.get('Category','')}")
    if re.search(r'\b(sock|socks|hosiery|tights|stockings|underwear|briefs|boxers|bra|bralette|lingerie|bikini|swimwear|swimsuit|pyjamas|pajamas|nightwear)\b', t): return None
    for role, pattern in ROLE_PATTERNS.items():
        if re.search(pattern, t): return role
    if norm(row.get('Category')) == 'footwear': return 'footwear'
    return None

def audience_of(row):
    a = norm(row.get('Audience'))
    if a in {'men','women','kids','unisex','unknown'}: return a
    t = norm(f"{row.get('Product_Name','')} {row.get('Subcategory','')}")
    if re.search(r"\b(women'?s|womens|ladies|female|woman)\b", t): return 'women'
    if re.search(r"\b(men'?s|mens|male|man)\b", t): return 'men'
    if re.search(r"\b(kids?|children|boys?|girls?|juniors?|youth)\b", t): return 'kids'
    if norm(row.get('Category')) in {'accessories','footwear'} and infer_role(row) in {'bag','footwear','accessory'}: return 'unisex'
    return 'unknown'
def audience_ok(row, audience, role=None):
    a = norm(audience)
    if not a or a == 'any': return True
    item_a = audience_of(row)
    if a == 'men':
        if item_a in {'women','kids'}: return False
        if role in {'upperwear','lowerwear','footwear','one_piece'}: return item_a in {'men','unisex'}
        return item_a in {'men','unisex'}
    if a == 'women':
        if item_a in {'men','kids'}: return False
        if role in {'upperwear','lowerwear','footwear','one_piece'}: return item_a in {'women','unisex'}
        return item_a in {'women','unisex'}
    if a == 'kids': return item_a == 'kids'
    if a == 'unisex': return item_a == 'unisex'
    return False

def parse_intent(message):
    t = norm(message)
    audience = 'any'
    if re.search(r"\b(men|men's|mens|for men|male|him|boyfriend|husband|father|dad)\b", t): audience='men'
    elif re.search(r"\b(women|women's|womens|for women|female|her|mother|mum|mom|wife|girlfriend)\b", t): audience='women'
    elif re.search(r"\b(kids?|children|child|son|daughter)\b", t): audience='kids'
    role = next((r for r,p in ROLE_PATTERNS.items() if re.search(p,t)), None)
    colour = None
    for c in ['black','white','navy','blue','grey','gray','beige','brown','green','red','pink','purple','yellow','orange','silver','gold']:
        if re.search(rf'\b{c}\b',t): colour='grey' if c=='gray' else c; break
    return {'audience':audience,'role':role,'colour':colour}

def filtered_by_metadata(df, audience='any', role=None, colour=None, max_price=None, include_unknown_audience=False):
    f=df.copy()
    if role:
        if '_role_key' in f: f=f[f['_role_key']==role]
        else: f=f[f.apply(infer_role,axis=1)==role]
    if audience and audience!='any':
        if '_audience_key' in f:
            allowed = {'men': {'men','unisex'}, 'women': {'women','unisex'}, 'kids': {'kids'}, 'unisex': {'unisex'}}.get(audience, set())
            f=f[f['_audience_key'].isin(allowed)]
        else: f=f[f.apply(lambda r: audience_ok(r,audience,role),axis=1)]
    if colour:
        colour_text = (f.get('Colour', pd.Series('', index=f.index)).fillna('').astype(str) + ' ' + f.get('Product_Name', pd.Series('', index=f.index)).fillna('').astype(str)).str.lower()
        text_match = colour_text.str.contains(re.escape(colour.lower()), regex=True)
        if 'Normalized_Colour' in f:
            norm_match = f['Normalized_Colour'].fillna('').astype(str).str.lower().eq(colour.lower())
            f=f[norm_match | text_match]
        else: f=f[text_match]
    prices=price_series(f); f=f[prices.notna() & (prices>0)]
    if max_price is not None: f=f[price_series(f)<=max_price]
    if 'Product_ID' in f: f=f.drop_duplicates('Product_ID')
    return f

def search_df(q=None,category=None,subcategory=None,colour=None,max_price=None,limit=24, audience='any', role=None):
    df=catalogue(); f=df.copy()
    if category and 'Category' in f: f=f[f['Category'].fillna('').astype(str).str.contains(category,case=False,regex=False)]
    if subcategory and 'Subcategory' in f: f=f[f['Subcategory'].fillna('').astype(str).str.contains(subcategory,case=False,regex=False)]
    intent=parse_intent(q or '')
    if role is None: role=intent['role']
    if audience=='any' and intent['audience']!='any': audience=intent['audience']
    if colour is None: colour=intent['colour']
    f=filtered_by_metadata(f,audience,role,colour,max_price)
    # Query terms rank candidates; terms don't all have to match, which avoids empty results for natural language.
    if q and q.strip() and not role:
        terms=[x for x in re.findall(r'[a-z0-9-]+',norm(q)) if len(x)>2 and x not in STOP and x not in {'men','mens','women','womens','male','female','mother','mum','mom','father','dad','him','her','black','white','navy','blue','grey','gray','beige','brown','green','red','pink','purple','yellow','orange','silver','gold','under','budget'}]
        if terms:
            hay=f.apply(text_for,axis=1)
            f=f.assign(_query_score=sum(hay.str.contains(re.escape(term),regex=True).astype(int) for term in terms))
            f=f[f['_query_score']>0].sort_values('_query_score',ascending=False,kind='stable').drop(columns=['_query_score'])
    elif q and q.strip() and role:
        # Role is a hard filter; product terms/brand names rank within eligible role.
        terms=[x for x in re.findall(r'[a-z0-9-]+',norm(q)) if len(x)>2 and x not in STOP and x not in {'men','mens','women','womens','male','female','mother','mum','mom','father','dad','him','her','black','white','navy','blue','grey','gray','beige','brown','green','red','pink','purple','yellow','orange','silver','gold','under','budget','bag','bags','handbag','handbags','shoe','shoes','trainers','trainer','dress','shirt','shirts','tops','top','jeans','jean','trousers','pants','boots','sneakers','sandal','sandals'}]
        if terms:
            hay=f.apply(text_for,axis=1); f=f.assign(_query_score=sum(hay.str.contains(re.escape(term),regex=True).astype(int) for term in terms)); f=f.sort_values('_query_score',ascending=False,kind='stable').drop(columns=['_query_score'])
    if 'Product_Name' in f and not q:
        f = f.sort_values('Product_Name', kind='stable')

    # Keep the full match count for API pagination/summary, but honour the
    # requested result limit in the DataFrame returned to every caller.
    total_matches = len(f)
    try:
        result_limit = max(1, int(limit))
    except (TypeError, ValueError):
        result_limit = 24
    result = f.head(result_limit).copy()
    result.attrs['total_matches'] = total_matches
    return result

def style_score(row, style, occasion, season, colour):
    t=text_for(row); s=norm(style+' '+occasion); score=0
    if colour and (norm(row.get('Normalized_Colour'))==colour or colour in norm(row.get('Colour'))): score+=3
    tags=norm(row.get('Style_Tags','')).replace('|',' ')
    if 'formal' in s or 'office' in s or 'wedding' in s:
        if any(x in t for x in ['blazer','trouser','loafer','heel','dress shoe','shirt','tailored']): score+=3
        if any(x in t for x in ['flip flop','jogger','hoodie','running shoe']): score-=4
    if 'casual' in s or 'weekend' in s or 'travel' in s:
        if any(x in t for x in ['trainer','jean','t-shirt','jumper','sneaker','hoodie']): score+=2
    if 'streetwear' in s and any(x in t for x in ['oversized','hoodie','trainer','sneaker','graphic']): score+=3
    if 'minimal' in s and any(x in t for x in ['plain','minimal','classic','black','white','neutral']): score+=1
    if season in ('winter','autumn') and any(x in t for x in ['coat','boot','jumper','knit','fleece']): score+=2
    if season=='summer' and any(x in t for x in ['linen','shorts','sandal','lightweight','vest']): score+=2
    if tags: score+=0.25
    return score

def groq_ready(): return bool(os.getenv('GROQ_API_KEY'))
async def groq_answer(system,user):
    key=os.getenv('GROQ_API_KEY')
    if not key: return None
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            r=await client.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {key}'},json={'model':GROQ_MODEL,'temperature':0.1,'messages':[{'role':'system','content':system},{'role':'user','content':user}]})
            r.raise_for_status(); return r.json()['choices'][0]['message']['content']
    except Exception: return None
async def gemini_image_answer(image_bytes,mime_type,prompt):
    key=os.getenv('GEMINI_API_KEY')
    if not key: return None
    url=f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}'
    payload={'contents':[{'parts':[{'text':prompt},{'inline_data':{'mime_type':mime_type,'data':base64.b64encode(image_bytes).decode()}}]}],'generationConfig':{'temperature':0.1}}
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r=await client.post(url,json=payload); r.raise_for_status(); return r.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception: return None

@app.get('/api/health')
def health():
    df = catalogue()
    source = str(df['_catalogue_source'].iloc[0]) if len(df) and '_catalogue_source' in df else (MASTER_PATH.name if MASTER_PATH.exists() else CLEAN_PATH.name if CLEAN_PATH.exists() else RAW_PATH.name)
    return {
        'status': 'ok',
        'catalogue_loaded': True,
        'catalogue_file': source,
        'catalogue_schema': 'products_master' if source == MASTER_PATH.name else 'legacy',
        'product_count': len(df),
        'columns_count': len(df.columns),
        'enriched_metadata': 'Outfit_Role' in df.columns and 'Audience' in df.columns,
        'needs_review_count': int(df['Needs_Review'].fillna(False).astype(bool).sum()) if 'Needs_Review' in df else int((df.get('Data_Quality_Status', pd.Series(index=df.index, dtype=object)) == 'review').sum()),
        'groq_configured': groq_ready(),
        'gemini_configured': bool(os.getenv('GEMINI_API_KEY')),
    }
@app.get('/api/catalogue/summary')
def summary():
    df=catalogue(); return {'product_count':len(df),'columns':list(df.columns),'categories':df['Category'].fillna('UNKNOWN').astype(str).value_counts().to_dict(),'unique_product_ids':int(df['Product_ID'].nunique()),'brands':int(df['Brand'].nunique()) if 'Brand' in df else 0,'audiences':df['Audience'].value_counts().to_dict() if 'Audience' in df else {},'outfit_roles':df['Outfit_Role'].value_counts().to_dict() if 'Outfit_Role' in df else {}}
@app.get('/api/products')
def products(q:Optional[str]=None,category:Optional[str]=None,subcategory:Optional[str]=None,colour:Optional[str]=None,max_price:Optional[float]=Query(default=None,ge=0),audience:Optional[str]=None,role:Optional[str]=None,limit:int=Query(default=24,ge=1,le=100)):
    f = search_df(q, category, subcategory, colour, max_price, limit, audience or 'any', role)
    return {'total_matches': int(f.attrs.get('total_matches', len(f))), 'returned': len(f), 'products': safe_products(f, limit)}
@app.get('/api/products/{product_id}')
def product_detail(product_id:str):
    df=catalogue(); f=df[df['Product_ID'].astype(str)==product_id]
    if f.empty: raise HTTPException(404,'Product ID not found in the catalogue.')
    return product_dict(f.iloc[0])

def choose_coherent_audience(df, budget, colour, style, occasion, season):
    # 'any' must not combine explicitly men's and women's garments in one outfit.
    candidates=[]
    for audience in ('men','women','unisex'):
        valid=filtered_by_metadata(df,audience,None,colour,budget)
        role_counts=valid['_role_key'].value_counts().to_dict() if '_role_key' in valid else valid.apply(infer_role,axis=1).value_counts().to_dict()
        role_counts={role:int(role_counts.get(role,0)) for role in ('upperwear','lowerwear','footwear')}
        complete=sum(role_counts[r]>0 for r in role_counts)
        total=sum(role_counts.values())
        candidates.append((complete,total,audience))
    candidates.sort(reverse=True)
    return candidates[0][2] if candidates and candidates[0][0] else 'unisex'

@app.post('/api/stylist/outfit')
def outfit(req:OutfitRequest):
    if req.audience not in {'any','men','women','kids','unisex'}: raise HTTPException(422,'Audience must be any, men, women, kids, or unisex.')
    df=catalogue()
    effective_audience = choose_coherent_audience(df,req.budget,req.colour,req.style,req.occasion,req.season) if req.audience=='any' else req.audience
    valid=filtered_by_metadata(df,effective_audience,None,req.colour,req.budget)
    core=['upperwear','lowerwear','footwear']; selected=[]; used=set(); remaining=float(req.budget); missing=[]
    for role in core:
        cand=filtered_by_metadata(valid,effective_audience,role,req.colour,remaining)
        if cand.empty:
            missing.append(role); continue
        cand=cand.copy(); cand['_score']=cand.apply(lambda r:style_score(r,req.style,req.occasion,req.season,req.colour),axis=1); cand['_price']=price_series(cand)
        # Balance relevance and value; don't simply select the cheapest item. Keep products from the requested audience only.
        cand['_value']=cand['_score'] - (cand['_price']/max(req.budget,1))*1.8
        cand=cand.sort_values(['_value','_score','_price'],ascending=[False,False,True],kind='stable')
        item=None
        for _,r in cand.iterrows():
            p=product_dict(r)
            if p.get('Product_ID') not in used: item=p; break
        if not item: missing.append(role); continue
        price=float(item['Selling_Price_GBP'])
        if price>remaining: missing.append(role); continue
        selected.append({'role':role,'product':item,'price_gbp':round(price,2)}); used.add(item['Product_ID']); remaining=round(remaining-price,2)
    if req.include_accessories and remaining>0:
        for role in ['bag','accessory']:
            cand=filtered_by_metadata(valid,effective_audience,role,req.colour,remaining)
            if cand.empty: continue
            cand=cand.copy(); cand['_score']=cand.apply(lambda r:style_score(r,req.style,req.occasion,req.season,req.colour),axis=1); cand['_price']=price_series(cand)
            cand=cand.sort_values(['_score','_price'],ascending=[False,True],kind='stable')
            for _,r in cand.iterrows():
                item=product_dict(r); price=float(item['Selling_Price_GBP'])
                if item.get('Product_ID') not in used and price<=remaining:
                    selected.append({'role':role,'product':item,'price_gbp':round(price,2)}); used.add(item['Product_ID']); remaining=round(remaining-price,2); break
    total=round(sum(x['price_gbp'] for x in selected),2)
    return {'occasion':req.occasion,'style':req.style,'audience':effective_audience,'requested_audience':req.audience,'season':req.season,'requested_budget_gbp':round(req.budget,2),'total_gbp':total,'remaining_budget_gbp':round(req.budget-total,2),'within_budget':total<=req.budget,'complete_core_outfit':not missing,'missing_roles':missing,'items':selected,'explanation':f'Products and prices come from the enriched catalogue. Selected audience: {effective_audience}. Audience and outfit roles are hard filters; style and colour coordination use transparent heuristics and are not a guarantee of perfect fashion compatibility.','ai_used':False}

@app.post('/api/assistant/ask')
async def assistant_ask(req:AskRequest):
    intent=parse_intent(req.message)
    f=search_df(req.message,max_price=req.budget,limit=60,audience=intent['audience'],role=intent['role'])
    # When role is explicit, never fall back to unrelated catalogue categories.
    records=safe_products(f,8)
    if not records:
        answer='I could not find a confident match in the catalogue for that product type and audience. Try another colour, brand, or a broader budget.'
        return {'answer':answer,'products':[],'grounded':True,'ai_used':False,'retrieval':'metadata filters + keyword ranking','intent':intent}
    context=[{'id':p.get('Product_ID'),'name':p.get('Product_Name'),'brand':p.get('Brand'),'category':p.get('Category'),'subcategory':p.get('Subcategory'),'colour':p.get('Colour'),'audience':p.get('Audience'),'role':p.get('Outfit_Role'),'price_gbp':p.get('Selling_Price_GBP')} for p in records]
    answer=await groq_answer('You are a luxury shopping assistant. The provided records are the only products you may recommend. Never invent or change IDs, names, prices, audience, product type, availability, materials or URLs. If the query asks for a product type, only discuss supplied records of that type. Say when the matches are limited.',f'Customer request: {req.message}\nBudget GBP: {req.budget}\nParsed intent: {json.dumps(intent)}\nCatalogue records JSON: {json.dumps(context,ensure_ascii=False)}')
    if not answer: answer=f'I found {len(records)} eligible catalogue match(es). Product facts below are from the supplied catalogue. Refine by brand, colour, or budget for a closer match.'
    return {'answer':answer,'products':records,'grounded':True,'ai_used':bool(answer and groq_ready()),'retrieval':'metadata filters + keyword ranking','intent':intent}

@app.post('/api/gifts/recommend')
async def gifts(req: GiftRequest):
    """Rank gift candidates using explicit customer preferences and catalogue evidence.

    This is intentionally rules-first: the LLM does not choose products or create facts.
    """
    recipient = norm(req.recipient)
    audience = norm(req.recipient_audience or 'any')
    if audience not in {'any', 'men', 'women', 'kids', 'unisex'}:
        audience = 'any'
    if audience == 'any':
        if re.search(r'\b(mother|mum|mom|wife|girlfriend|sister|daughter|her)\b', recipient): audience = 'women'
        elif re.search(r'\b(father|dad|husband|boyfriend|brother|son|him)\b', recipient): audience = 'men'

    df = catalogue()
    if 'Giftable' in df:
        f = df[df['Giftable'].fillna(False).astype(bool)].copy()
    else:
        f = df.copy()
    f = filtered_by_metadata(f, audience, None, None, req.budget)

    # Exclude explicitly unwanted categories/terms. This is a hard rule, not a score penalty.
    avoid_terms = [norm(x) for x in re.split(r'[,;|]', req.avoid or '') if norm(x)]
    if avoid_terms and not f.empty:
        searchable = f.apply(text_for, axis=1)
        for term in avoid_terms:
            f = f.loc[~searchable.loc[f.index].str.contains(re.escape(term), case=False, na=False)].copy()
            searchable = f.apply(text_for, axis=1)

    if f.empty:
        return {
            'recipient': req.recipient, 'occasion': req.occasion, 'sentiment': req.sentiment,
            'budget_gbp': req.budget, 'products': [],
            'recommendation': 'I could not find eligible catalogue gifts matching the budget, audience and exclusions together. Try a broader gift type, remove an exclusion, or increase the budget.',
            'gift_message': '', 'ai_used': False, 'grounded': True, 'audience_filter': audience,
            'match_summary': 'No product met all hard constraints.'
        }

    def tokens(value):
        return [w for w in re.findall(r'[a-z0-9]+', norm(value)) if len(w) > 2 and w not in STOP]

    interest_terms = tokens(req.interests)
    style_terms = tokens(req.style_preference)
    occasion_terms = tokens(req.occasion)
    sentiment_terms = tokens(req.sentiment)
    preferred = norm(req.preferred_category)
    category_aliases = {
        'beauty_fragrance': ['beauty', 'fragrance', 'perfume', 'skincare', 'cosmetic'],
        'jewellery': ['jewellery', 'jewelry', 'necklace', 'bracelet', 'earring', 'ring'],
        'bag': ['bag', 'handbag', 'tote', 'purse', 'clutch'],
        'accessory': ['accessory', 'accessories', 'watch', 'belt', 'scarf', 'sunglasses', 'wallet'],
        'gift_set': ['gift set', 'set', 'bundle', 'duo'],
        'home_gift': ['home', 'candle', 'diffuser', 'homeware'],
    }
    priority = {'gift_set': 5, 'beauty_fragrance': 4, 'jewellery': 4, 'bag': 3, 'accessory': 2, 'home_gift': 2}
    ranked = []
    for _, row in f.iterrows():
        record = row.to_dict()
        hay = norm(text_for(row))
        score = 0
        reasons = []
        # Interest match is strongest because it is an explicit signal from the customer.
        matched_interests = [w for w in interest_terms if w in hay]
        if matched_interests:
            score += 8 * len(matched_interests)
            reasons.append('matches interest: ' + ', '.join(matched_interests[:4]))
        matched_style = [w for w in style_terms if w in hay]
        if matched_style:
            score += 4 * len(matched_style)
            reasons.append('matches style: ' + ', '.join(matched_style[:3]))
        matched_occasion = [w for w in occasion_terms if w in hay]
        if matched_occasion:
            score += 3 * len(matched_occasion)
            reasons.append('occasion-related catalogue wording')
        matched_sentiment = [w for w in sentiment_terms if w in hay]
        if matched_sentiment:
            score += 2 * len(matched_sentiment)
        gift_cat = norm(row.get('Gift_Category', ''))
        category_text = norm(f"{row.get('Category','')} {row.get('Subcategory','')} {row.get('Product_Type_Normalized','')} {gift_cat}")
        if preferred != 'any':
            aliases = category_aliases.get(preferred, [preferred.replace('_', ' ')])
            if any(alias in category_text for alias in aliases):
                score += 12
                reasons.append('matches preferred gift type')
            else:
                score -= 5
        score += priority.get(gift_cat, 0)
        price = pd.to_numeric(pd.Series([row.get('Selling_Price_GBP')]), errors='coerce').iloc[0]
        # Gentle budget fit: favour thoughtful use of budget without simply selecting the cheapest item.
        if pd.notna(price) and req.budget > 0:
            ratio = float(price) / float(req.budget)
            if 0.45 <= ratio <= 0.90:
                score += 2
            elif ratio > 0.90:
                score += 1
        record['Gift_Match_Score'] = round(float(score), 2)
        record['Gift_Match_Reason'] = '; '.join(reasons) if reasons else 'eligible catalogue gift; limited preference-specific evidence'
        record['_gift_score'] = score
        record['_gift_price'] = float(price) if pd.notna(price) else float('inf')
        ranked.append(record)

    # Stable tie-break: score first, then a budget-aware price preference, then product name.
    ranked.sort(key=lambda x: (-x['_gift_score'], abs((x['_gift_price'] / req.budget) - 0.7) if x['_gift_price'] != float('inf') else 9, str(x.get('Product_Name', ''))))
    top = ranked[:12]
    recs = []
    for item in top:
        item.pop('_gift_score', None); item.pop('_gift_price', None)
        recs.append(product_dict(item))

    top_names = ', '.join(str(p.get('Product_Name') or 'catalogue item') for p in recs[:3])
    preference_summary = []
    if interest_terms: preference_summary.append('interests: ' + ', '.join(interest_terms[:5]))
    if style_terms: preference_summary.append('style: ' + ', '.join(style_terms[:5]))
    if preferred != 'any': preference_summary.append('gift type: ' + preferred.replace('_', ' '))
    summary = ('Ranked against ' + '; '.join(preference_summary) + '. ' if preference_summary else 'No specific interests or style were supplied, so ranking relies on eligible gift metadata. ')
    recommendation = f'Here are catalogue products within £{req.budget:,.2f}, filtered for the selected audience and exclusions. Best starting points: {top_names}. ' + summary + 'Product facts and prices are taken from the supplied catalogue; the catalogue cannot confirm the recipient’s personal tastes.'
    gift_message = f'Wishing you a wonderful {req.occasion.lower()}. I chose this with {req.sentiment.lower()} in mind and hope it makes you feel appreciated.'
    return {
        'recipient': req.recipient, 'occasion': req.occasion, 'sentiment': req.sentiment,
        'budget_gbp': req.budget, 'products': recs, 'recommendation': recommendation,
        'gift_message': gift_message, 'ai_used': False, 'grounded': True,
        'audience_filter': audience, 'match_summary': summary,
        'ranking_method': 'deterministic preference scoring; explicit exclusions and budget are hard filters'
    }

@app.post('/api/products/ask')
async def ask_product(req:ProductQuestion):
    df=catalogue(); f=df[df['Product_ID'].astype(str)==req.product_id.strip()]
    if f.empty: raise HTTPException(404,'Product ID not found in the catalogue.')
    p=product_dict(f.iloc[0]); fields=['Product_ID','Product_Name','Brand','Category','Subcategory','Product_Type_Normalized','Colour','Material','Fit','Selling_Price_GBP','Ticket_Price_GBP','Discount_Percentage','Available_Sizes','Audience','Outfit_Role','Style_Tags','Occasion_Tags','Season_Tags','Product_Description','Care_Instructions','Country_Of_Manufacture','Activity','Activity_Group']
    context={k:p.get(k) for k in fields}
    answer=await groq_answer('Answer only from the provided catalogue fields. If material, care, fit, stock, delivery or another fact is absent, explicitly say the catalogue does not provide it. Never guess. Product ID is authoritative.',f'Product record: {json.dumps(context,ensure_ascii=False)}\nQuestion: {req.question}')
    return {'product':p,'answer':answer or 'Verified catalogue details: '+json.dumps(context,ensure_ascii=False)+'\nNo additional material, care, stock or delivery information is available in the supplied catalogue.','grounded':True,'ai_used':bool(answer)}

@app.post('/api/matching/image')
async def image_matching(image:UploadFile=File(...),budget:Optional[float]=Form(default=None),notes:str=Form(default='Find similar or complementary fashion products'),audience:str=Form(default='any')):
    if image.content_type not in ['image/jpeg','image/png','image/webp']: raise HTTPException(415,'Upload a JPG, PNG or WEBP image.')
    data=await image.read()
    if len(data)>8*1024*1024: raise HTTPException(413,'Image must be smaller than 8 MB.')
    description=await gemini_image_answer(data,image.content_type,'Describe visible fashion product types, dominant colours, patterns and style in searchable terms. Do not identify people. Separate observable attributes from uncertain guesses. User matching request: '+notes)
    generic_notes = norm(notes) in {'find similar or complementary fashion products','find similar products','match this image','find similar items'}
    if not description and generic_notes:
        return {'visual_description':'Image analysis is unavailable because Gemini is not configured or the request failed.','query_used':notes,'products':[],'ai_used':False,'grounded':True,'message':'No image-based matches were generated. Configure Gemini or add a specific written description such as “black leather handbag” to search the catalogue.'}
    query=description or notes
    intent=parse_intent(notes+' '+(description or ''))
    if audience=='any' and intent['audience']!='any': audience=intent['audience']
    role=intent['role']
    # Never return arbitrary products when image interpretation/search fails.
    f=search_df(query,max_price=budget,limit=80,audience=audience,role=role)
    products=safe_products(f,16)
    return {'visual_description':description or 'Gemini is not configured or image analysis failed; results use the written request only.','query_used':query,'products':products,'ai_used':bool(description),'grounded':True,'message':None if products else 'No confident catalogue match found. Add a more specific note (e.g. black handbag, men’s navy trainers) or configure Gemini image analysis.'}

@app.get('/api/merchandising/summary')
def merchandising_summary():
    df=catalogue(); prices=price_series(df); valid=prices.dropna(); cat=df['Category'].fillna('Unknown').value_counts().to_dict(); brands=df['Brand'].fillna('Unknown').value_counts().head(12).to_dict() if 'Brand' in df else {}; sub=df['Subcategory'].fillna('Unknown').value_counts().head(15).to_dict() if 'Subcategory' in df else {}
    discounts=pd.to_numeric(df.get('Discount_Percentage',pd.Series(index=df.index,dtype=float)),errors='coerce')
    return {'product_count':len(df),'unique_brands':int(df['Brand'].nunique()) if 'Brand' in df else 0,'categories':cat,'top_brands':brands,'top_subcategories':sub,'price_stats':{'min':round(float(valid.min()),2) if len(valid) else None,'median':round(float(valid.median()),2) if len(valid) else None,'mean':round(float(valid.mean()),2) if len(valid) else None,'max':round(float(valid.max()),2) if len(valid) else None},'discount_stats':{'mean_discount_percent':round(float(discounts.mean()),2) if discounts.notna().any() else None,'products_with_discount':int((discounts>0).sum())},'missing_image_count':int(df['Image_URL'].isna().sum()) if 'Image_URL' in df else int(df.get('Main_Image_URL',pd.Series(index=df.index,dtype=object)).isna().sum()),'audiences':df['Audience'].value_counts().to_dict() if 'Audience' in df else {},'outfit_roles':df['Outfit_Role'].value_counts().to_dict() if 'Outfit_Role' in df else {}}
@app.post('/api/occasion/recommend')
def occasion_recommend(req:OutfitRequest): return outfit(req)
@app.post('/api/wardrobe/validate')
def wardrobe_validate(req:WardrobeRequest):
    df=catalogue(); wanted=set(str(x) for x in req.product_ids); f=df[df['Product_ID'].astype(str).isin(wanted)]
    return {'count':len(f),'products':safe_products(f,100),'missing_ids':sorted(wanted-set(f['Product_ID'].astype(str)))}
@app.get('/api/metadata/audit')
def metadata_audit():
    df=catalogue()
    return {'source':str(df['_catalogue_source'].iloc[0]) if len(df) and '_catalogue_source' in df else 'catalogue','rows':len(df),'audience_counts':df['Audience'].value_counts().to_dict() if 'Audience' in df else {},'role_counts':df['Outfit_Role'].value_counts().to_dict() if 'Outfit_Role' in df else {},'needs_review_count':int(df['Needs_Review'].fillna(False).astype(bool).sum()) if 'Needs_Review' in df else None,'sample_review':safe_products(df[df.get('Needs_Review',pd.Series(False,index=df.index)).fillna(False).astype(bool)],10)}
