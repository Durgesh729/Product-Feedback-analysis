import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_and_evaluate_unseen_test(data_path="data/cleaned_multi_domain_reviews.csv"):
    print("--- STARTING PHASE 10: 135-EXAMPLE UNSEEN MANUAL TEST & QUALITATIVE DENTAL TEST ---")

    domains = ['electronics', 'personal care', 'kitchen', 'apparel', 'automotive', 'beauty', 'home', 'sports', 'grocery']

    raw_examples = {
        'electronics': {
            'Positive': [
                "The battery life on these noise canceling headphones is phenomenal.",
                "Crisp audio display and smooth refresh rate make this monitor exceptional.",
                "Fast wireless charging pad that works reliably every single night.",
                "The soundstage on these studio monitors exceeded all my expectations.",
                "Seamless Bluetooth connectivity and incredible bass performance."
            ],
            'Negative': [
                "The screen cracked after two days and the touch response is completely unresponsive.",
                "Terrible Wi-Fi reception and it constantly disconnects during streaming.",
                "Overheats within ten minutes of light usage and shuts down repeatedly.",
                "The charging port loose pin broke on the first week of purchase.",
                "Low quality speaker with distorted audio and static noise."
            ],
            'Neutral': [
                "The tablet comes with a standard USB cable and user manual in the box.",
                "Screen resolution is 1080p and measures 15.6 inches diagonally.",
                "It features two HDMI ports and one optical audio output jack.",
                "The device operates on 5 volts DC input power supply.",
                "Includes a basic plastic carrying case and warranty documentation."
            ]
        },
        'personal care': {
            'Positive': [
                "This electric toothbrush leaves my teeth feeling dentist clean every morning.",
                "Super quiet hair dryer that dries my hair in half the usual time.",
                "Very smooth shave with zero skin irritation or razor burn.",
                "Great battery runtime on this beard trimmer and easy to clean heads.",
                "Gentle water flosser with adjustable pressure settings that works wonderfully."
            ],
            'Negative': [
                "The razor blades dull after just one use and pull on facial hair.",
                "The heating element on this curling iron melted the plastic tip.",
                "Battery stopped holding charge after three charges on this shaver.",
                "Extremely loud motor and vibration makes it uncomfortable to hold.",
                "The attachment heads snap off easily and feel flimsy."
            ],
            'Neutral': [
                "The package includes four replacement brush heads and a travel pouch.",
                "This hair trimmer runs on two AA batteries and has a LED indicator.",
                "The device dimensions are 7 inches by 2 inches by 1.5 inches.",
                "Contains stainless steel blades and a washable waterproof housing.",
                "Operates at two distinct speed levels selected via a side toggle switch."
            ]
        },
        'kitchen': {
            'Positive': [
                "Blends frozen fruits into smooth smoothies in seconds effortlessly.",
                "Heavy duty cast iron skillet that retains heat evenly across the surface.",
                "Nonstick coating works amazingly well and cleans up with a simple wipe.",
                "Sharp chef knife that cuts through meats and vegetables effortlessly.",
                "The pressure cooker seals perfectly and cooks meals in record time."
            ],
            'Negative': [
                "The nonstick layer started peeling into our food after two weeks of hand washing.",
                "Glass lid shattered under normal stovetop heat producing dangerous shards.",
                "Blender motor smoked and burnt out on the first attempt to crush ice.",
                "Coffee maker leaks water from the bottom reservoir onto the counter.",
                "Can opener gears slipped and jammed completely on the first tin."
            ],
            'Neutral': [
                "The kettle holds 1.7 liters of liquid and features auto shutoff.",
                "Made of 304 grade stainless steel with a brushed metallic finish.",
                "Includes three measuring cups and two silicone spatulas.",
                "Dishwasher safe cookware set rated for up to 400 degrees Fahrenheit.",
                "Measures 12 inches long by 8 inches wide and weighs 2 pounds."
            ]
        },
        'apparel': {
            'Positive': [
                "Super comfortable cotton fabric with perfect fit and great stitching quality.",
                "Warm fleece lining that keeps me cozy during cold winter runs.",
                "Durable denim jeans that retain their color and shape after multiple washes.",
                "Breathable mesh material that keeps feet cool during long workouts.",
                "Elegant dress fit that matches the size chart exactly as described."
            ],
            'Negative': [
                "Shirt shrunk two sizes after a single cold water wash.",
                "The zipper tore away from the fabric on the first wear.",
                "Colors faded completely into a dull grey after one machine wash.",
                "Itchy cheap material that feels uncomfortable against the skin.",
                "Seams unravelled around the armpits after light casual walking."
            ],
            'Neutral': [
                "The sweater is composed of 80 percent cotton and 20 percent polyester.",
                "Available in sizes Small through Extra Large in neutral gray color.",
                "Features two side pockets and a standard front button closure.",
                "Washing instructions recommend cold machine wash with mild detergent.",
                "Garment weight is approximately 350 grams with ribbed cuffs."
            ]
        },
        'automotive': {
            'Positive': [
                "Bright LED headlight bulbs that significantly improve night driving visibility.",
                "Heavy duty floor mats that trap dirt and fit the car footwells perfectly.",
                "Smooth wiper blades that clear heavy rain without squeaking or streaking.",
                "Compact tire inflator that pumped up my flat tire quickly and quietly.",
                "High quality car wax that leaves a deep glossy protective shine."
            ],
            'Negative': [
                "Oil filter threading was stripped causing a severe engine leak.",
                "Car battery failed completely after only one month of normal driving.",
                "Brake pads produced unbearable squealing noise and excessive dust.",
                "Dash cam video recorded corrupted unplayable files during driving.",
                "Seat covers ripped along the elastic seams while being installed."
            ],
            'Neutral': [
                "Standard 12 volt car accessory plug with an inline 10 amp fuse.",
                "Fits standard 15 inch wheel rims and includes mounting hardware.",
                "Synthetic motor oil meeting SAE 5W-30 viscosity specifications.",
                "The package contains two microfiber cleaning cloths and one applicator pad.",
                "Designed for OBD2 protocol vehicle diagnostic ports."
            ]
        },
        'beauty': {
            'Positive': [
                "Hydrating face moisturizer that absorbs quickly without feeling greasy.",
                "Vibrant lipstick color that stays intact all day without smudging.",
                "Restorative hair serum that eliminated frizz and added healthy shine.",
                "Gentle cleansing foam that leaves skin feeling refreshed and soft.",
                "Excellent foundation coverage that blends smoothly with natural skin tone."
            ],
            'Negative': [
                "Caused severe skin redness and allergic breakout within hours.",
                "Mascara clumpy application that flaked off into eyes throughout the day.",
                "Strong unpleasant chemical perfume scent that triggered severe headaches.",
                "Nail polish peeled off in full sheets within hours of application.",
                "Eye shadow container arrived completely crushed with powder spilled everywhere."
            ],
            'Neutral': [
                "Formulated without parabens or sulfates in a 50ml glass jar.",
                "Product contains hyaluronic acid and vitamin C as active ingredients.",
                "Net weight is 1.7 ounces with a pump dispenser nozzle.",
                "Expiration date is stamped on the bottom of the outer cardboard box.",
                "Apply twice daily to clean skin using light upward strokes."
            ]
        },
        'home': {
            'Positive': [
                "Ultra soft bed sheets that feel like staying in a luxury hotel.",
                "Powerful vacuum cleaner that picked up all stubborn pet hair effortlessly.",
                "Sturdy bookshelf that assembled easily with clear concise instructions.",
                "Blackout curtains that block 100 percent of morning sunlight effectively.",
                "Quiet room air purifier that noticeably reduced dust and allergies."
            ],
            'Negative': [
                "Desk leg collapsed under light weight causing computer monitor to fall.",
                "Vacuum cleaner lost all suction power on the third day of use.",
                "Curtain rod bent under the weight of light fabric drapery.",
                "Cheap particle board emitted foul chemical fumes for weeks.",
                "Mattress sagged significantly in the middle after two weeks."
            ],
            'Neutral': [
                "The floor lamp stands 60 inches tall and uses a standard E26 bulb.",
                "Constructed from engineered wood with an oak veneer finish.",
                "Set includes one fitted sheet, one flat sheet, and two pillowcases.",
                "Operates on 120 volt AC household electrical power outlet.",
                "Dimensions are 36 inches long by 18 inches deep by 30 inches high."
            ]
        },
        'sports': {
            'Positive': [
                "Durable yoga mat with superior floor grip that does not slip at all.",
                "High quality resistance bands that withstand heavy tension workouts easily.",
                "Comfortable running shoes with plush cushioning and great arch support.",
                "Sturdy mountain bike helmet with easy adjustment dial and light weight.",
                "Waterproof camping tent that stayed completely dry through a rainstorm."
            ],
            'Negative': [
                "Resistance band snapped during first stretch hitting my arm hard.",
                "Basketball lost air pressure within two hours of inflation repeatedly.",
                "Trekking pole lock slipped causing it to collapse while hiking uphill.",
                "Goggles fogged up instantly and leaked pool water into eyes.",
                "Dumbbell coating cracked exposing rusted iron underneath upon arrival."
            ],
            'Neutral': [
                "The water bottle has a 32 ounce capacity with volume markers.",
                "Yoga block measures 9 inches by 6 inches by 4 inches in size.",
                "Made from high density EVA foam material weighing 0.5 pounds.",
                "Includes a nylon carrying strap and mesh storage bag.",
                "Suitable for indoor gym exercise and outdoor fitness training."
            ]
        },
        'grocery': {
            'Positive': [
                "Rich dark roast coffee beans with incredible fresh aroma and flavor.",
                "Crisp delicious organic granola with wholesome nuts and dried berries.",
                "Authentic extra virgin olive oil with rich pepper notes and great taste.",
                "Delicious dark chocolate bars with smooth texture and rich cocoa content.",
                "Fresh herbal tea bags that brew a soothing comforting cup every time."
            ],
            'Negative': [
                "Package arrived completely expired by three months with moldy contents.",
                "Stale rock hard cookies that tasted completely rancid upon opening.",
                "Canned goods arrived crushed and leaking inside the shipping box.",
                "Bland flavorless soup mix containing excessive salt and artificial taste.",
                "Bag had a tear spilling loose grains all over the outer box."
            ],
            'Neutral': [
                "Contains 12 individually wrapped single serve snack packets.",
                "Net weight 16 ounces certified organic by USDA guidelines.",
                "Nutrition facts report 150 calories and 3 grams of sugar per serving.",
                "Store in a cool dry place away from direct sunlight exposure.",
                "Ingredients include whole oats, almonds, honey, and natural vanilla."
            ]
        }
    }

    # Build dataframe for 135 benchmark examples
    rows = []
    for domain, sent_dict in raw_examples.items():
        for expected_sent, text_list in sent_dict.items():
            for text in text_list:
                rows.append({
                    'domain': domain,
                    'expected_sentiment': expected_sent,
                    'review_text': text,
                    'is_qualitative': False
                })

    # Add the 1 qualitative test case (Dental/Interdental brush sentence)
    dental_sentence = "The angle provided in this interdental brush is not accurate. It hurts the gums. The bristle pin length is very short for pre molar and molar teeth. So difficult clean teeth from one side. Poor quality plastic is used."
    rows.append({
        'domain': 'personal care',
        'expected_sentiment': 'Negative',
        'review_text': dental_sentence,
        'is_qualitative': True
    })

    df_unseen = pd.DataFrame(rows)

    # Verify 0% text overlap with training data
    df_train_clean = pd.read_csv(data_path)
    train_texts = set(df_train_clean['review_body'].fillna('').str.strip().str.lower())
    train_titles = set(df_train_clean['review_title'].fillna('').str.strip().str.lower())

    overlap_count = 0
    for idx, r in df_unseen.iterrows():
        t_low = r['review_text'].strip().lower()
        if t_low in train_texts or t_low in train_titles:
            overlap_count += 1

    assert overlap_count == 0, f"0% Overlap constraint violated! Found {overlap_count} overlapping sentences."
    print("VERIFIED: 0% Text Overlap between Unseen Manual Test sentences and Dataset!")

    # Save to data/unseen_multidomain_test.csv
    os.makedirs("data", exist_ok=True)
    df_unseen.to_csv("data/unseen_multidomain_test.csv", index=False)
    print(f"Saved 136 unseen test cases (135 benchmark + 1 qualitative) to data/unseen_multidomain_test.csv")

    # Run predictions using final trained model
    with open("models/multi_domain_sentiment_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/multi_domain_tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    X_unseen_vec = vec.transform(df_unseen['review_text'])
    preds = clf.predict(X_unseen_vec)
    probs = clf.predict_proba(X_unseen_vec)

    classes = list(clf.classes_)
    neg_idx = classes.index('Negative')
    neu_idx = classes.index('Neutral')
    pos_idx = classes.index('Positive')

    df_unseen['predicted_sentiment'] = preds
    df_unseen['prob_Negative'] = probs[:, neg_idx]
    df_unseen['prob_Neutral'] = probs[:, neu_idx]
    df_unseen['prob_Positive'] = probs[:, pos_idx]

    # Evaluate 135 benchmark examples
    df_bm = df_unseen[~df_unseen['is_qualitative']]
    acc_bm = accuracy_score(df_bm['expected_sentiment'], df_bm['predicted_sentiment'])
    p_m, r_m, f1_bm, _ = precision_recall_fscore_support(df_bm['expected_sentiment'], df_bm['predicted_sentiment'], average='macro')

    # Qualitative dental test case result
    dental_row = df_unseen[df_unseen['is_qualitative']].iloc[0]

    lines = []
    lines.append("\n==================================================")
    lines.append("   135-EXAMPLE UNSEEN MANUAL SANITY TEST REPORT   ")
    lines.append("==================================================")
    lines.append(f"Total Benchmark Examples: {len(df_bm)} (9 Domains x 3 Sentiments x 5 Examples)")
    lines.append(f"Text Overlap with Training Data: 0% (VERIFIED)")
    lines.append(f"Benchmark Accuracy: {acc_bm*100:.2f}% ({acc_bm:.4f})")
    lines.append(f"Benchmark Macro F1: {f1_bm:.4f}")

    lines.append("\nPER-DOMAIN UNSEEN BENCHMARK ACCURACY & MACRO F1:")
    lines.append(f"{'Domain':<18} | {'Accuracy':<8} | {'Macro F1':<8}")
    lines.append("-" * 42)
    for d in sorted(domains):
        d_df = df_bm[df_bm['domain'] == d]
        d_acc = accuracy_score(d_df['expected_sentiment'], d_df['predicted_sentiment'])
        _, _, d_f1, _ = precision_recall_fscore_support(d_df['expected_sentiment'], d_df['predicted_sentiment'], average='macro', zero_division=0)
        lines.append(f"{d:<18} | {d_acc*100:6.2f}%  | {d_f1:<8.4f}")

    lines.append("\n--- QUALITATIVE INTERDENTAL BRUSH TEST RESULT ---")
    lines.append(f"Review Text: \"{dental_row['review_text']}\"")
    lines.append(f"Domain: {dental_row['domain']}")
    lines.append(f"Expected Sentiment: {dental_row['expected_sentiment']}")
    lines.append(f"Predicted Sentiment: {dental_row['predicted_sentiment']}")
    lines.append(f"Predicted Class Probabilities:")
    lines.append(f"   - Negative: {dental_row['prob_Negative']:.4f}")
    lines.append(f"   - Neutral:  {dental_row['prob_Neutral']:.4f}")
    lines.append(f"   - Positive: {dental_row['prob_Positive']:.4f}")

    report_str = "\n".join(lines)
    print(report_str)

    # Append to reports/MULTI_DOMAIN_MODEL_EVALUATION.txt
    with open("reports/MULTI_DOMAIN_MODEL_EVALUATION.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + report_str)

    # Save detailed prediction table
    os.makedirs("outputs", exist_ok=True)
    df_unseen.to_csv("outputs/unseen_multidomain_test_results.csv", index=False)

    print("Appended unseen sanity test report to reports/MULTI_DOMAIN_MODEL_EVALUATION.txt")
    return df_unseen

if __name__ == "__main__":
    generate_and_evaluate_unseen_test()
