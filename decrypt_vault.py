import os
import json
import nacl.public
import nacl.utils
from datetime import datetime

def decrypt_and_generate_dashboard():
    vault_dir = "secure_vault"
    output_dir = "decrypted_audio"
    key_path = "developer_private.key"
    reports_file = "feedback_reports.jsonl"
    dashboard_file = "reports_dashboard.html"

    print("=== Secure Vault Decryption & Dashboard Utility ===")

    if not os.path.exists(key_path):
        print(f"[ERROR] Private key '{key_path}' not found!")
        print("This script must be run on the secure offline machine holding the private key.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load the Private Key
    with open(key_path, "rb") as kf:
        private_key = nacl.public.PrivateKey(kf.read())
    unseal_box = nacl.public.SealedBox(private_key)

    # 1. Parse JSONL Log File
    reports = []
    if os.path.exists(reports_file):
        with open(reports_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    reports.append(json.loads(line))
                    
    # Sort from newest to oldest
    reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    # 2. Decrypt & Group by Session Date
    success = 0
    for report in reports:
        if "utterance_id" not in report or "audio_file" not in report:
            continue
            
        enc_path = report["audio_file"]
        if not os.path.exists(enc_path):
            continue
            
        # Group by Date (creates folders like: decrypted_audio/2026-06-12/)
        date_str = report["timestamp"].split("T")[0]
        session_dir = os.path.join(output_dir, date_str)
        os.makedirs(session_dir, exist_ok=True)
        
        out_filename = f"{report['utterance_id']}.wav"
        out_path = os.path.join(session_dir, out_filename)
        
        # Save relative path for the HTML dashboard
        report["decrypted_path"] = f"{output_dir}/{date_str}/{out_filename}"

        if not os.path.exists(out_path): # Only decrypt if not already decrypted
            try:
                with open(enc_path, "rb") as f:
                    encrypted_data = f.read()
                decrypted_data = unseal_box.decrypt(encrypted_data)
                with open(out_path, "wb") as f:
                    f.write(decrypted_data)
                print(f" [OK] Decrypted: {out_filename} into {date_str}/")
                success += 1
            except Exception as e:
                print(f" [FAIL] Could not decrypt {enc_path}: {e}")

    # 3. Generate HTML Dashboard
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Audio Telemetry Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d0f14; color: #fff; padding: 30px; margin: 0; }
        h1 { border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 30px; font-weight: 400; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; background: rgba(255,255,255,0.02); border-radius: 8px; overflow: hidden; }
        th, td { padding: 15px; text-align: left; border-bottom: 1px solid #222; }
        th { background: #1a1d24; color: #888; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }
        tr:last-child td { border-bottom: none; }
        tr:hover { background: rgba(255,255,255,0.05); }
        .correction { color: #ff4d6d; font-weight: 500; }
        .source { color: #e0e0e0; }
        .translated { color: #aaa; font-style: italic; }
        .uid { font-family: 'Consolas', monospace; color: #00d4ff; background: rgba(0, 212, 255, 0.1); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .time { color: #666; font-size: 12px; }
        audio { height: 35px; width: 220px; outline: none; border-radius: 30px; }
        audio::-webkit-media-controls-panel { background-color: #1a1d24; color: #fff; }
    </style>
</head>
<body>
    <h1>🎙️ Audio Feedback Telemetry</h1>
    <table>
        <tr>
            <th>Date / Time</th>
            <th>ID</th>
            <th>Original STT (Source)</th>
            <th>Translated Output</th>
            <th>User Correction</th>
            <th>Audio Playback</th>
        </tr>"""

    for r in reports:
        raw_dt = r.get("timestamp", "")[:19]
        dt = raw_dt.replace("T", "<br><span style='color:#444'>") + "</span>" if raw_dt else "N/A"
        uid = r.get("utterance_id", "N/A")
        src = r.get("source_text", "")
        tgt = r.get("translated_text", "")
        corr = r.get("correction", "")
        path = r.get("decrypted_path", "")
        
        audio_html = f'<audio controls src="{path}"></audio>' if path else "<span style='color:#444'>No Audio</span>"
        
        html += f"""
        <tr>
            <td class="time">{dt}</td>
            <td><span class="uid">{uid}</span></td>
            <td class="source">{src}</td>
            <td class="translated">{tgt}</td>
            <td class="correction">{corr}</td>
            <td>{audio_html}</td>
        </tr>"""

    html += """
    </table>
</body>
</html>"""

    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"\n[DONE] Decrypted {success} new files.")
    print(f"[SUCCESS] Generated dashboard! Open '{dashboard_file}' in your web browser.")

if __name__ == "__main__":
    decrypt_and_generate_dashboard()
