import re
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DECISION_THRESHOLD = 0.57

class URLFeatureExtractor(BaseEstimator, TransformerMixin):

    # ── URL shorteners ────────────────────────────────────────────────────────
    URL_SHORTENERS = {
        'bit.ly', 't.co', 'tinyurl.com', 'goo.gl', 'ow.ly',
        'short.link', 'buff.ly', 'rebrand.ly', 'cutt.ly', 'is.gd',
        't.ly', 'rb.gy', 'shorturl.at', 'tiny.cc', 'bl.ink',
        'snip.ly', 'clck.ru', 'qr.ae', 'po.st', 'su.pr'
    }

    # ── High-abuse TLDs ───────────────────────────────────────────────────────
    # Sources: Spamhaus, APWG, threat-intel reports 2024-2025
    HIGH_RISK_TLDS = {
        # Freenom — historically free, overwhelmingly abused
        'tk', 'ml', 'ga', 'cf', 'gq',
        # Cheap new gTLDs with confirmed high abuse rates
        'xyz', 'top', 'club', 'online', 'site', 'website',
        'store', 'shop', 'live', 'click', 'link', 'world',
        # Threat-intel confirmed high-abuse
        'cfd', 'cyou', 'sbs', 'bar', 'buzz', 'cam', 'icu', 'vip'
    }

    # ── Known brands for subdomain impersonation detection ───────────────────
    # Curated — not auto-derived from Tranco.
    # Tranco gives root domains (paypal.com), this gives brand strings (paypal)
    # for matching paypal.evil.com patterns.
    KNOWN_BRANDS = {
        'paypal', 'apple', 'google', 'microsoft', 'amazon', 'netflix',
        'facebook', 'instagram', 'linkedin', 'twitter', 'chase',
        'wellsfargo', 'bankofamerica', 'citibank', 'hsbc', 'dropbox',
        'adobe', 'steam', 'ebay', 'dhl', 'fedex', 'docusign',
        'onedrive', 'sharepoint', 'quickbooks', 'roblox', 'tiktok',
        'yahoo', 'outlook', 'office', 'coinbase', 'binance', 'stripe'
    }

    # ── Phishing keywords in path ─────────────────────────────────────────────
    # Catches combosquatting where brand is in path not domain:
    # e.g. evil.xyz/paypal/login/verify/account
    PHISHING_PATH_KEYWORDS = {
        'login', 'signin', 'sign-in', 'verify', 'verification',
        'secure', 'account', 'update', 'confirm', 'banking',
        'payment', 'invoice', 'billing', 'authenticate', 'auth',
        'recover', 'unlock', 'restore', 'suspend', 'alert',
        'validate', 'credential', 'password', 'reactivate'
    }
    
    def __init__(self, trusted_domains=None):
        self.trusted_domains = trusted_domains

    def fit(self, X, y=None):
        """Learn TLD probability table from training URLs."""
        tld_counts = {}
        total      = 0
        for url in X:
            tld = self._get_tld(str(url))
            tld_counts[tld] = tld_counts.get(tld, 0) + 1
            total += 1
        self.tld_prob_table_ = {k: v / total for k, v in tld_counts.items()}
        return self

    def transform(self, X, y=None):
        return pd.DataFrame([self._extract(str(url)) for url in X])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_tld(self, url):
        try:
            parsed = urlparse(url if url.startswith('http') else 'http://' + url)
            return parsed.netloc.split('.')[-1].split(':')[0].lower()
        except Exception:
            return ''

    def _get_root_domain(self, url):
        try:
            parsed = urlparse(url if url.startswith('http') else 'http://' + url)
            netloc = parsed.netloc.lower().split(':')[0]
            parts  = netloc.split('.')
            return '.'.join(parts[-2:]) if len(parts) >= 2 else netloc
        except Exception:
            return ''

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract(self, url):
        f = {}

        try:
            parsed = urlparse(url if url.startswith('http') else 'http://' + url)
            domain = parsed.netloc.lower()
            tld    = domain.split('.')[-1].split(':')[0] if '.' in domain else ''
            subs   = [s for s in domain.split('.')[:-2] if s] \
                     if domain.count('.') > 1 else []
        except Exception:
            parsed = urlparse('')   # safe empty object — all attributes return ''
            domain = tld = ''
            subs   = []

        # ── Group 1: URL structure ────────────────────────────────────────────
        f['URLLength']     = len(url)
        f['DomainLength']  = len(domain)
        f['TLDLength']     = len(tld)
        f['NoOfSubDomain'] = len(subs)
        f['IsHTTPS']       = 1 if url.startswith('https') else 0
        f['IsDomainIP']    = 1 if re.match(
            r'^\d{1,3}(\.\d{1,3}){3}$', domain.split(':')[0]) else 0

        # ── Group 2: Path & query ─────────────────────────────────────────────
        # Separates domain signal from path signal.
        # Phishing pattern: short domain + long suspicious path
        f['PathLength']  = len(parsed.path)
        f['QueryLength'] = len(parsed.query)

        # ── Group 3: Character statistics ────────────────────────────────────
        letters = sum(c.isalpha() for c in url)
        digits  = sum(c.isdigit() for c in url)

        f['NoOfLettersInURL'] = letters
        f['LetterRatioInURL'] = letters / max(len(url), 1)
        f['NoOfDigitsInURL']  = digits
        f['DigitRatioInURL']  = digits  / max(len(url), 1)
        f['NoOfEqualsInURL']  = url.count('=')
        f['NoOfQMarkInURL']   = url.count('?')
        f['NoOfDotsInURL']    = url.count('.')
        f['NoOfDotsInDomain'] = domain.count('.')

        # ── Group 4: Entropy ──────────────────────────────────────────────────
        url_chars = re.sub(r'[^a-zA-Z0-9]', '', url)
        if len(url_chars) > 1:
            freq    = Counter(url_chars)
            total   = len(url_chars)
            entropy = -sum((c / total) * math.log2(c / total)
                           for c in freq.values())
            f['URLCharProb']          = round(1 - (entropy / 5.17), 6)
            f['CharContinuationRate'] = sum(
                1 for i in range(len(url_chars) - 1)
                if url_chars[i].isalpha() == url_chars[i + 1].isalpha()
            ) / max(len(url_chars) - 1, 1)
        else:
            f['URLCharProb']          = 0.0
            f['CharContinuationRate'] = 0.0

        # ── Group 5: Domain reputation ────────────────────────────────────────
        tld_table = getattr(self, 'tld_prob_table_', {})
        trusted   = self.trusted_domains if self.trusted_domains is not None else set()

        f['TLDLegitimateProb']   = tld_table.get(tld, 0.0)
        f['IsTrustedDomain']     = 1 if self._get_root_domain(url) in trusted else 0
        f['IsURLShortener']      = 1 if self._get_root_domain(url) in \
                                   self.__class__.URL_SHORTENERS else 0
        f['IsHighRiskTLD']       = 1 if tld in self.__class__.HIGH_RISK_TLDS else 0
        f['NoOfHyphensInDomain'] = domain.count('-')

        # ── Group 6: Attack pattern signals ──────────────────────────────────

        # @ symbol — browser ignores everything before @
        # http://legit.com@evil.com routes to evil.com, not legit.com
        f['HasAtSymbol'] = 1 if '@' in url else 0

        # Phishing keywords in path — catches combosquatting where brand
        # is in path not domain: evil.xyz/paypal/login/verify
        f['HasPhishingKeywordInPath'] = int(
            any(kw in parsed.path.lower()
                for kw in self.__class__.PHISHING_PATH_KEYWORDS)
        )

        # Brand in subdomain — catches paypal.evil.com, apple.phishing.xyz
        # IsTrustedDomain correctly returns 0 for these — without this
        # feature the model has no explicit signal for the pattern
        subdomain_str = '.'.join(domain.split('.')[:-2]).lower()
        f['HasBrandInSubdomain'] = int(
            any(brand in subdomain_str
                for brand in self.__class__.KNOWN_BRANDS)
        )

        # ── Group 7: Interaction features ────────────────────────────────────
        # Validated: HighRiskTLD_x_Hyphens → -68 FN at baseline
        # Tested and dropped: HighRiskTLD_x_QMark, Shortener_x_URLLength
        f['HighRiskTLD_x_Hyphens'] = f['IsHighRiskTLD'] * f['NoOfHyphensInDomain']

        return f


def predict_with_override(pipeline, urls, trusted_domains,
                           url_shorteners=None, threshold=DECISION_THRESHOLD):
    if url_shorteners is None:
        url_shorteners = URLFeatureExtractor.URL_SHORTENERS

    extractor  = pipeline.named_steps['extractor']
    model      = pipeline.named_steps['model']
    classes    = list(model.classes_)
    phish_col  = classes.index(0)

    n           = len(urls)
    preds       = np.ones(n, dtype=int)
    phish_probs = np.zeros(n, dtype=float)
    layer       = ['whitelist'] * n
    unknown_idx = []

    for i, url in enumerate(urls):
        root = extractor._get_root_domain(url)
        if root in url_shorteners:
            unknown_idx.append(i)
            layer[i] = 'ml_shortener_bypass'
        elif root in trusted_domains:
            preds[i]       = 1
            phish_probs[i] = 0.0
            layer[i]       = 'whitelist'
        else:
            unknown_idx.append(i)
            layer[i] = 'ml_model'

    if unknown_idx:
        unknown_urls  = [urls[i] for i in unknown_idx]
        proba         = pipeline.predict_proba(unknown_urls)
        unknown_probs = proba[:, phish_col]

        for j, i in enumerate(unknown_idx):
            prob           = unknown_probs[j]
            phish_probs[i] = round(float(prob), 4)
            preds[i]       = 0 if prob >= threshold else 1
    return preds, phish_probs, layer
