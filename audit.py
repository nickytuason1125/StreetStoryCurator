import os, sys, cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from lightweight_analyzer import LightweightStreetScorer

def create_dummy_images(folder, count=10):
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        # Random noise + shapes to simulate street photos
        img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        cv2.rectangle(img, (50,50), (200,200), (255,255,255), 3)
        cv2.imwrite(os.path.join(folder, f"test_img_{i}.jpg"), img)
    return [os.path.join(folder, f"test_img_{i}.jpg") for i in range(count)]

def run_audit():
    test_dir = "cache/audit_test_images"
    create_dummy_images(test_dir)

    analyzer = LightweightStreetScorer()
    print("🧪 Running lightweight audit test...")

    results = analyzer.analyze_folder(test_dir, preset="Magnum Editor")
    score, paths = 0, []
    for path, data in results:
        score += data["score"]
        paths.append(path)
        assert isinstance(data["score"], float), "Score must be float"
        assert len(data["embedding"]) > 0, "Embedding missing"
        print(f"  ✅ {os.path.basename(path)}: {data['score']} ({data['grade']})")

    seq, rationale = analyzer.sequence_story(results, target=3)
    print(f"📊 Avg Score: {score/len(results):.2f} | Sequence: {len(seq)} frames")
    print("✅ Audit PASSED. Pipeline is stable and ready for UI.")

    # Cleanup
    for f in os.listdir(test_dir): os.remove(os.path.join(test_dir, f))
    os.rmdir(test_dir)

if __name__ == "__main__":
    run_audit()
