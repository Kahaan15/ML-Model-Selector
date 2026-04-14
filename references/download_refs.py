import re
import os
import time
import urllib.request
import urllib.error
import urllib.parse
import json

def extract_references(tex_file):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract bibitems
    pattern = re.compile(r'\\bibitem\{(b\d+)\}\s*(.*?)(?=\\bibitem|\Z)', re.DOTALL)
    matches = pattern.findall(content)
    
    refs = []
    for ref_id, text in matches:
        text = text.replace('\n', ' ').strip()
        doi_match = re.search(r'doi:\s*([^\s,]+)', text)
        url_match = re.search(r'Available:\s*(https?://[^\s,]+)', text)
        
        doi = doi_match.group(1).rstrip('.') if doi_match else None
        url = url_match.group(1).rstrip('.') if url_match else None
        
        # Simple heuristic to extract first author and year
        author_match = re.search(r'^([A-Za-z\.\s-]+),', text)
        author = author_match.group(1).split(',')[0].split()[-1] if author_match else "Unknown"
        year_match = re.search(r'(19\d{2}|20\d{2})', text)
        year = year_match.group(1) if year_match else "XXXX"
        
        filename = f"{author}{year}_{ref_id}.pdf"
        
        refs.append({
            'id': ref_id,
            'text': text,
            'doi': doi,
            'url': url,
            'filename': filename
        })
        
    return refs

def download_pdf(url, filename, out_dir):
    try:
        print(f"Attempting to download {filename} from {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            if b'%PDF' in content[:10]:
                with open(os.path.join(out_dir, filename), 'wb') as f:
                    f.write(content)
                print(f"Successfully downloaded {filename}")
                return True
            else:
                print(f"Failed or not a PDF at {url}")
                return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def check_unpaywall(doi, filename, out_dir):
    # Try unpaywall API
    email = "test@example.com"
    api_url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={email}"
    try:
        print(f"Checking Unpaywall for DOI: {doi}")
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('is_oa'):
                best_oa = data.get('best_oa_location', {})
                if best_oa and best_oa.get('url_for_pdf'):
                    pdf_url = best_oa['url_for_pdf']
                    return download_pdf(pdf_url, filename, out_dir)
                elif best_oa and best_oa.get('url'):
                    # Try the generic URL just in case
                    pdf_url = best_oa['url']
                    return download_pdf(pdf_url, filename, out_dir)
    except urllib.error.HTTPError as e:
        print(f"Unpaywall NOT FOUND for {doi} (HTTP {e.code})")
    except Exception as e:
        print(f"Unpaywall error for {doi}: {e}")
    return False

def create_placeholder(filename, text, out_dir):
    # Some older papers or books don't have open PDFs. We will generate a markdown reference file instead.
    md_filename = filename.replace('.pdf', '.md')
    print(f"Creating reference marker {md_filename}")
    with open(os.path.join(out_dir, md_filename), 'w', encoding='utf-8') as f:
        f.write(f"# Reference Info: {filename}\n\n")
        f.write(f"{text}\n\n")
        f.write("> Auto-download of this specific PDF failed or paper is fully paywalled. User instructed to download all references, generating reference marker for completeness.")

def main():
    tex_file = '/Users/kartikadhonde/Documents/ML-Model-Selector/main_ml_v2.tex'
    out_dir = '/Users/kartikadhonde/Documents/ML-Model-Selector/references'
    
    os.makedirs(out_dir, exist_ok=True)
    refs = extract_references(tex_file)
    
    print(f"Found {len(refs)} references in {tex_file}")
    
    for r in refs:
        # Check if already downloaded
        pdf_path = os.path.join(out_dir, r['filename'])
        if os.path.exists(pdf_path):
            print(f"Already possess: {r['filename']}")
            continue
            
        success = False
        if r['url'] and r['url'].endswith('.pdf'):
            success = download_pdf(r['url'], r['filename'], out_dir)
            
        if not success and r['doi']:
            success = check_unpaywall(r['doi'], r['filename'], out_dir)
            
        if not success and r['url'] and 'arxiv' in r['url']:
            # Construct arxiv pdf url
            pdf_url = r['url'].replace('/abs/', '/pdf/') + '.pdf'
            success = download_pdf(pdf_url, r['filename'], out_dir)
            
        if not success:
            # Create a placeholder document so EVERY reference has a representation in the folder
            create_placeholder(r['filename'], r['text'], out_dir)
            
        time.sleep(0.5) # Rate limit

if __name__ == '__main__':
    main()
