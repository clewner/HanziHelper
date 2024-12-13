
from paddleocr import PaddleOCR
from easyocr import Reader
from pypinyin import pinyin, Style, load_phrases_dict
from PIL import Image
from collections import Counter
import numpy as np
import opencc

# Constants
OVERLAP_THRESHOLD_PERCENT = 1  # Percentage overlap required for merging
CLOSENESS_THRESHOLD = 40  # Threshold for grouping bounding boxes into rows

# Helper function to check if a string contains Chinese characters
def contains_chinese(text):
    """
    Check if the text contains any Chinese characters.
    """
    if not isinstance(text, str):
        return False
    
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def contains_alphabet(text):
    """
    Check if the text contains any alphabetical characters (a-z, A-Z).
    Returns True if it contains at least one alphabet character, False otherwise.
    """
    if not isinstance(text, str):
        return False
    
    # Use any() for efficiency - stops checking once it finds an alphabet character
    return any(char.isalpha() for char in text)

def is_all_chinese_or_punctuation(text):
    """
    Check if a string contains only Chinese characters and/or punctuation.
    Returns True if text consists entirely of Chinese characters and punctuation,
    False if it contains any other characters (like letters or numbers).
    """
    if not isinstance(text, str):
        return False
    
    for char in text:
        # Skip punctuation and whitespace
        if char.isspace() or char in '.,;()[]{}/-、。，；（）「」【】':
            continue
        # If we find any non-Chinese character (except those we skipped above)
        if not '\u4e00' <= char <= '\u9fff':
            return False
    return True

def get_isolated_chinese_chars(text):
        """Find all single Chinese characters that are surrounded by non-Chinese characters or string boundaries"""
        if not isinstance(text, str):
            return []
        
        isolated_chars = []
        last_was_chinese = False
        
        # Add a space at the end to handle last character properly
        text = text + ' '
        
        for i, char in enumerate(text[:-1]):  # Exclude the added space
            is_chinese = '\u4e00' <= char <= '\u9fff'
            next_is_chinese = '\u4e00' <= text[i + 1] <= '\u9fff'
            
            if is_chinese:
                if not last_was_chinese and not next_is_chinese:
                    # Character is surrounded by non-Chinese or string boundaries
                    isolated_chars.append((char, i))
            
            last_was_chinese = is_chinese
        
        return isolated_chars

# Function to crop image and convert to numpy array
def crop_image(img_path, bbox):
    image = Image.open(img_path).convert("RGB")
    min_x = int(min([point[0] for point in bbox]))
    max_x = int(max([point[0] for point in bbox]))
    min_y = int(min([point[1] for point in bbox]))
    max_y = int(max([point[1] for point in bbox]))
    cropped_image = image.crop((min_x, min_y, max_x, max_y))
    return np.array(cropped_image)

# Function to calculate intersection area
def calculate_intersection_area(box1, box2):
    x1 = max(box1['min_x'], box2['min_x'])
    y1 = max(box1['min_y'], box2['min_y'])
    x2 = min(box1['max_x'], box2['max_x'])
    y2 = min(box1['max_y'], box2['max_y'])
    if x1 >= x2 or y1 >= y2:
        return 0
    return (x2 - x1) * (y2 - y1)

# Function to check if boxes should merge
def boxes_should_merge(box1, box2, tolerance=1):
    """
    Determines if two bounding boxes should be merged based on their position and content.
    
    Args:
        box1 (dict): First bounding box with keys 'min_x', 'max_x', 'min_y', 'max_y', 'text'
        box2 (dict): Second bounding box with same keys as box1
        tolerance (int): Maximum pixel distance to consider boxes as "touching" (default: 1)
        
    Returns:
        bool: True if boxes should be merged, False otherwise
    """
    # Validate input
    required_keys = ['min_x', 'max_x', 'min_y', 'max_y', 'text']
    for box in [box1, box2]:
        if not all(key in box for key in required_keys):
            raise KeyError(f"Box missing required keys. Must have: {required_keys}")
    
    # Check if either text contains english characters
    if is_all_chinese_or_punctuation(box1['text']) and is_all_chinese_or_punctuation(box2['text']):
        return False
    
    
    # Check for x-axis overlap or touching
    x_touching = (
        abs(box1['max_x'] - box2['min_x']) <= tolerance or  # box1 touches box2 from left
        abs(box2['max_x'] - box1['min_x']) <= tolerance     # box2 touches box1 from left
    )
    x_overlap = (
        box1['min_x'] <= box2['max_x'] and 
        box1['max_x'] >= box2['min_x']
    )
    
    # Check for y-axis overlap or touching
    y_touching = (
        abs(box1['max_y'] - box2['min_y']) <= tolerance or  # box1 touches box2 from top
        abs(box2['max_y'] - box1['min_y']) <= tolerance     # box2 touches box1 from top
    )
    y_overlap = (
        box1['min_y'] <= box2['max_y'] and 
        box1['max_y'] >= box2['min_y']
    )
    
    # Boxes should merge if they overlap or touch in both axes
    should_merge = (x_overlap or x_touching) and (y_overlap or y_touching)
    
    # Debug output
    """if should_merge:
        print(f"Merging boxes: '{box1['text']}' and '{box2['text']}'")
    else:
        print(f"Not merging boxes: '{box1['text']}' and '{box2['text']}'")"""
    
    return should_merge

def boxes_should_merge_phase2(box1, box2, row_info, tolerance=1, y_tolerance=40):

    """
    Check if boxes should merge based on row alignment.
    
    Args:
        box1: First box
        box2: Second box
        row_info: Dictionary containing:
            - row_midpoints: List of row y-midpoints (only for rows with multiple items)
            - box_to_row: Dictionary mapping box indices to their row indices
        tolerance: X-axis tolerance
        y_tolerance: Y-axis tolerance (larger than phase 1)
    """
    # Basic position validation
    required_keys = ['min_x', 'max_x', 'min_y', 'max_y', 'text']
    for box in [box1, box2]:
        if not all(key in box for key in required_keys):
            raise KeyError(f"Box missing required keys. Must have: {required_keys}")
    
    # Check if boxes are too far apart in Y direction
    box1_mid_y = (box1['min_y'] + box1['max_y']) / 2
    box2_mid_y = (box2['min_y'] + box2['max_y']) / 2

    
    # Check X-axis overlap/proximity using midpoints
    box1_mid_x = (box1['max_x'] - box1['min_x'])/2 + box1['min_x']
    box2_mid_x = (box2['max_x'] - box2['min_x'])/2 + box2['min_x']
    
    if not (box1_mid_x <= box2['max_x'] and box1_mid_x >= box2['min_x'] and 
            box2_mid_x <= box1['max_x'] and box2_mid_x >= box1['min_x']):
        return False
    


    
    # Calculate merged midpoint
    merged_min_y = min(box1['min_y'], box2['min_y'])
    merged_max_y = max(box1['max_y'], box2['max_y'])
    merged_mid_y = (merged_min_y + merged_max_y) / 2

    if(("不但···" in box1['text'] and "而且·.·" in box2['text']) or ("不但···" in box2['text'] and "而且·.·" in box1['text'])):
        print("found")
        print(box1['min_y'])
        print(box1['max_y'])
        print(box2['min_y'])
        print(box2['max_y'])
        print(min(box1['min_y'], box2['min_y']) - max(box1['max_y'], box2['max_y']))

    if abs(max(box1['min_y'], box2['min_y']) - min(box1['max_y'], box2['max_y'])) >= 40:
        return False

    print(box1['text'])
    print(box2['text'])
    

    # Find distances to valid row midpoints
    if not row_info['row_midpoints']:  # No valid rows to compare to
        return False
        
    # Calculate distances
    current_dist1 = min(abs(box1_mid_y - mid) for mid in row_info['row_midpoints'])
    current_dist2 = min(abs(box2_mid_y - mid) for mid in row_info['row_midpoints'])
    merged_dist = min(abs(merged_mid_y - mid) for mid in row_info['row_midpoints'])
    
    # Merge if it would bring boxes closer to a valid row
    return merged_dist < min(current_dist1, current_dist2)

def get_row_information(rows):
    """Convert row data into format needed for phase 2 merging."""
    row_info = {
        'row_midpoints': [],
        'box_to_row': {}
    }
    
    # Calculate midpoints only for rows with multiple items
    for row_idx, row in enumerate(rows):
        if len(row) > 1:
            # Calculate row midpoint excluding single boxes we're considering merging
            y_values = []
            for bbox_tuple in row:
                bbox = bbox_tuple[0]  # Get the bbox part
                points = bbox[0]  # Get the points
                ys = [point[1] for point in points]
                y_values.extend(ys)
            
            if y_values:
                row_midpoint = sum(y_values) / len(y_values)
                row_info['row_midpoints'].append(row_midpoint)
        
        # Map each box to its row
        for bbox_tuple in row:
            box_id = id(bbox_tuple[0])  # Use bbox object id as identifier
            row_info['box_to_row'][box_id] = row_idx
    
    return row_info

def merge_bounding_boxes(aabbs):
    merged = []
    skip_indices = set()
    for i in range(len(aabbs)):
        if i in skip_indices:
            continue
        box1 = aabbs[i]
        merged_box = box1.copy()
        for j in range(i + 1, len(aabbs)):
            if j in skip_indices:
                continue
            box2 = aabbs[j]
            if boxes_should_merge(merged_box, box2):
                merged_box['min_x'] = min(merged_box['min_x'], box2['min_x'])
                merged_box['max_x'] = max(merged_box['max_x'], box2['max_x'])
                merged_box['min_y'] = min(merged_box['min_y'], box2['min_y'])
                merged_box['max_y'] = max(merged_box['max_y'], box2['max_y'])
                merged_box['text'] += ' ' + box2['text']
                merged_box['confidence'] = (merged_box['confidence'] + box2['confidence']) / 2
                skip_indices.add(j)
        merged.append(merged_box)
    return merged

def merge_bounding_boxes_phase2(aabbs, row_info):
    print("MERGE PHASE 2")
    """Second phase merging - row-aware for Chinese text"""
    merged = []
    skip_indices = set()
    for i in range(len(aabbs)):
        if i in skip_indices:
            continue
        box1 = aabbs[i]
        merged_box = box1.copy()
        for j in range(i + 1, len(aabbs)):
            if j in skip_indices:
                continue
            box2 = aabbs[j]
            if boxes_should_merge_phase2(merged_box, box2, row_info):
                merged_box['min_x'] = min(merged_box['min_x'], box2['min_x'])
                merged_box['max_x'] = max(merged_box['max_x'], box2['max_x'])
                merged_box['min_y'] = min(merged_box['min_y'], box2['min_y'])
                merged_box['max_y'] = max(merged_box['max_y'], box2['max_y'])
                merged_box['text'] += ' ' + box2['text']
                merged_box['confidence'] = (merged_box['confidence'] + box2['confidence']) / 2
                skip_indices.add(j)
        merged.append(merged_box)
    return merged

# Step to handle rows with one less item
def handle_incomplete_rows_with_not_found(rows, mode_length):
    """
    Handle rows that have either one less or one more element than the mode length.
    For rows with one less: adds a "Not found" placeholder in the appropriate position
    For rows with one more: removes the element that least matches expected column positions
    """
    # Get reference rows with the mode length
    reference_rows = [row for row in rows if len(row) == mode_length]
    if not reference_rows:
        raise ValueError("No reference rows with mode length found.")

    # Calculate average x-coordinates for each column in the reference rows
    avg_x_positions = []
    for col_index in range(mode_length):
        x_positions = [
            bbox[0][0][0] for row in reference_rows for bbox in row[col_index:col_index + 1]
        ]
        avg_x_positions.append(sum(x_positions) / len(x_positions))

    adjusted_rows = []
    for row in rows:
        if len(row) == mode_length:
            adjusted_rows.append(row)
            
        elif len(row) == mode_length - 1:
            # Handle rows with one missing element
            row_x_positions = [bbox[0][0][0] for bbox in row]

            # Determine the missing index by comparing x-positions
            missing_index = None
            for i, avg_x in enumerate(avg_x_positions):
                if i >= len(row_x_positions) or abs(avg_x - row_x_positions[i]) > CLOSENESS_THRESHOLD:
                    missing_index = i
                    break

            # Default to the last position if not identified
            if missing_index is None:
                missing_index = len(row_x_positions)

            # Create a placeholder for "Not found"
            placeholder_bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]  # Dummy bounding box
            placeholder = [placeholder_bbox, ("Not found", 0.0)]

            # Insert the placeholder at the determined index
            new_row = row[:missing_index] + [placeholder] + row[missing_index:]
            adjusted_rows.append(new_row)
            
        elif len(row) == mode_length + 1:
            # Handle rows with one extra element
            row_x_positions = [bbox[0][0][0] for bbox in row]
            
            # For each element in the row, calculate how far it is from any expected column position
            element_deviations = []
            for i, x_pos in enumerate(row_x_positions):
                # Find distance to closest expected column position
                min_deviation = float('inf')
                for ref_x in avg_x_positions:
                    deviation = abs(x_pos - ref_x)
                    min_deviation = min(min_deviation, deviation)
                element_deviations.append((i, min_deviation))
            
            # Sort by deviation to find the element that's most "out of place"
            element_deviations.sort(key=lambda x: x[1], reverse=True)
            index_to_remove = element_deviations[0][0]
            
            # Create new row without the most deviant element
            new_row = row[:index_to_remove] + row[index_to_remove + 1:]
            adjusted_rows.append(new_row)
            
        else:
            # Rows with lengths too far from the mode are discarded
            continue

    return adjusted_rows



def strip_tone_marks(pinyin_str):
    """Remove tone marks from pinyin string"""
    # Replace tone marked vowels with regular vowels
    tone_marks = {
        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v'
    }
    result = ''
    for char in pinyin_str:
        result += tone_marks.get(char, char)
    return result



# Main processing function
def process_image(img_path, heteronym_callback=None, is_traditional=True):
    # Initialize OCR models
    if is_traditional:
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Keep existing settings for traditional
        easy_ocr = Reader(['ch_tra', 'en'])  # Keep existing settings for traditional
        converter = opencc.OpenCC('s2tw.json')

    else:
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Same for simplified
        easy_ocr = Reader(['ch_sim', 'en'])  # Change to simplified for EasyOCR

    # Step 1: Detection with PaddleOCR
    paddle_result = paddle_ocr.ocr(img_path, cls=True)

    firstimageresult = paddle_result[0]


    #pypinyin translates 夾 into jia
    load_phrases_dict({'夾': [['lái']]})

    # Step 2: Replace Chinese text with EasyOCR output
    processed_results = []
    for line in paddle_result[0]:
        bbox, (text, confidence) = line
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            cropped_image = crop_image(img_path, bbox)
            easy_result = easy_ocr.readtext(cropped_image, detail=0)
            print("PADDLE:")
            print(text)
            print(confidence)
            print("EASY:")
            #print(easy_result[0])
            
            
            if easy_result:
                if len(easy_result[0]) > len(text) or confidence < 0.90:
                    text = easy_result[0]
                    confidence = 1.0
                else:
                    print("no")
            
            if(is_traditional):
                text = converter.convert(text)  # 漢字

        processed_results.append([bbox, (text, confidence)])


    # Step 3: Compute Axis-Aligned Bounding Boxes (AABBs)
    aabbs = []
    for bbox in processed_results:
        if isinstance(bbox[0], list) and isinstance(bbox[1], tuple):  # Ensure structure is correct
            points = bbox[0]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            aabbs.append({
                'bbox': bbox,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'text': bbox[1][0],
                'confidence': bbox[1][1]
            })

    # Step 4: Merge overlapping bounding boxes
    merged = merge_bounding_boxes(aabbs)

    # Step 5: Convert merged boxes back into PaddleOCR format
    result_merged = []
    for box in merged:
        min_x, max_x = box['min_x'], box['max_x']
        min_y, max_y = box['min_y'], box['max_y']
        new_bbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        result_merged.append([new_bbox, (box['text'], box['confidence'])])

    # Step 6: Group bounding boxes into rows and columns
    bboxes_with_avg = []
    for bbox in result_merged:
        if isinstance(bbox[0], list) and isinstance(bbox[1], tuple):  # Ensure structure is correct
            points = bbox[0]
            ys = [point[1] for point in points]
            xs = [point[0] for point in points]
            avg_y = sum(ys) / len(ys)
            avg_x = sum(xs) / len(xs)
            bboxes_with_avg.append((bbox, avg_y, avg_x))

    # Sort by vertical position (y-axis)
    bboxes_with_avg.sort(key=lambda x: x[1])


    # Group bounding boxes into rows
    rows = []
    if bboxes_with_avg:  # Ensure there are bounding boxes to process
        current_row = [bboxes_with_avg[0]]
        rows.append(current_row)

        for bbox_with_avg in bboxes_with_avg[1:]:
            bbox, avg_y, avg_x = bbox_with_avg
            last_avg_y = current_row[-1][1]
            if abs(avg_y - last_avg_y) < CLOSENESS_THRESHOLD:
                current_row.append(bbox_with_avg)
            else:
                current_row = [bbox_with_avg]
                rows.append(current_row)

    # Sort each row by horizontal position (x-axis)
    for row in rows:
        row.sort(key=lambda x: x[2])

    rows_info = get_row_information(rows)

    # Convert row data back to AABB format for phase 2 merging
    aabbs_phase2 = []
    for row in rows:
        for bbox_tuple in row:
            bbox = bbox_tuple[0]
            points = bbox[0]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            aabbs_phase2.append({
                'bbox': bbox,
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys),
                'text': bbox[1][0],
                'confidence': bbox[1][1]
            })

    # Perform phase 2 merging
    merged_phase2 = merge_bounding_boxes_phase2(aabbs_phase2, rows_info)
    result_merged_phase2 = []
    for box in merged_phase2:
        min_x, max_x = box['min_x'], box['max_x']
        min_y, max_y = box['min_y'], box['max_y']
        new_bbox = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]
        result_merged_phase2.append([new_bbox, (box['text'], box['confidence'])])

    # Regroup into rows with averages (same as before)
    bboxes_with_avg = []
    for bbox in result_merged_phase2:
        if isinstance(bbox[0], list) and isinstance(bbox[1], tuple):
            points = bbox[0]
            ys = [point[1] for point in points]
            xs = [point[0] for point in points]
            avg_y = sum(ys) / len(ys)
            avg_x = sum(xs) / len(xs)
            bboxes_with_avg.append((bbox, avg_y, avg_x))

    # Sort and regroup into rows
    bboxes_with_avg.sort(key=lambda x: x[1])
    rows = []
    if bboxes_with_avg:
        current_row = [bboxes_with_avg[0]]
        rows.append(current_row)

        for bbox_with_avg in bboxes_with_avg[1:]:
            bbox, avg_y, avg_x = bbox_with_avg
            last_avg_y = current_row[-1][1]
            if abs(avg_y - last_avg_y) < CLOSENESS_THRESHOLD:
                current_row.append(bbox_with_avg)
            else:
                current_row = [bbox_with_avg]
                rows.append(current_row)

    # Sort each row by x position
    for row in rows:
        row.sort(key=lambda x: x[2])

    # Clean rows for further processing
    rows_cleaned = [[bbox[0] for bbox in row] for row in rows]
    print("Rows Before Cleaning:")
    for row in rows_cleaned:
        print([bbox[1][0] for bbox in row if isinstance(bbox[1], tuple)])  # Ensure proper indexing

    # Step 7: Detect and handle header row
    row_lengths = [len(row) for row in rows_cleaned]
    if not row_lengths:  # Ensure there are rows to process
        raise ValueError("No rows detected in the image.")

    length_counts = Counter(row_lengths)
    mode_length = length_counts.most_common(1)[0][0]

    # Filter rows with mode length or one less
    filtered_rows = [row for row in rows_cleaned if len(row) in [mode_length - 1, mode_length, mode_length + 1]]

    # Identify header row and ensure it's preserved
    header_row = next((row for row in filtered_rows if len(row) == mode_length), None)
    if not header_row:
        raise ValueError("No header row with mode length found.")
    column_texts = [bbox[1][0].lower().strip() for bbox in header_row if isinstance(bbox[1], tuple)]
    print("Header Row Detected:", column_texts)

    # Handle incomplete rows
    filtered_rows = handle_incomplete_rows_with_not_found(filtered_rows, mode_length)

    # Ensure header row remains unchanged

    column_texts = [bbox[1][0].lower().strip() for bbox in header_row if isinstance(bbox[1], tuple)]
    print("Header Row Detected:", column_texts)

    # Ensure the correct columns are identified
    def find_column_index(column_names, target_name):
        for idx, name in enumerate(column_names):
            if name.strip().lower() == target_name:
                return idx
        return -1

    pinyin_col_index = find_column_index(column_texts, "pinyin")
    word_col_index = find_column_index(column_texts, "word")

    if pinyin_col_index == -1 and column_texts.count("word") > 1:
        print("Warning: 'pinyin' not found. Assigning second 'word' column to 'pinyin'.")
        pinyin_col_index = column_texts.index("word", 1)

    if pinyin_col_index == -1 or word_col_index == -1:
        raise ValueError("Required columns ('pinyin', 'word') could not be identified.")

    # Step 8: Process rows and translate words to Pinyin
    final_table = []
    firstrow = True
    for row in filtered_rows:
        row_texts = [bbox[1][0] for bbox in row if isinstance(bbox[1], tuple)]
        
        if not firstrow:  # Skip header row
            word_text = row_texts[word_col_index]
            
            # Find all isolated Chinese characters
            isolated_chars = get_isolated_chinese_chars(word_text)
            
            if isolated_chars:
                # Process each isolated character for heteronyms
                pinyin_parts = []
                last_pos = 0
                
                for char, pos in isolated_chars:
                    # Add text before this character
                    if pos > last_pos:
                        prefix_text = word_text[last_pos:pos]
                        if prefix_text:
                            prefix_pinyin = pinyin(prefix_text, style=Style.TONE)
                            pinyin_parts.extend(item[0] for item in prefix_pinyin)
                    
                    # Process the isolated character
                    readings = pinyin(char, style=Style.TONE, heteronym=True)[0]
                    if len(readings) > 1:
                        # Strip tone marks and check if all base pinyin are the same
                        stripped_readings = [strip_tone_marks(r) for r in readings]
                        if len(set(stripped_readings)) == 1:  # All readings are same base pinyin
                            if heteronym_callback:
                                pinyin_text = heteronym_callback(char, readings)
                            else:
                                pinyin_text = readings[0]
                        else:
                            pinyin_text = readings[0]
                    else:
                        pinyin_text = readings[0]
                    
                    pinyin_parts.append(pinyin_text)
                    last_pos = pos + 1
                
                # Add remaining text
                if last_pos < len(word_text):
                    remaining_text = word_text[last_pos:]
                    if remaining_text:
                        remaining_pinyin = pinyin(remaining_text, style=Style.TONE)
                        pinyin_parts.extend(item[0] for item in remaining_pinyin)
                
                pinyin_text = ''.join(pinyin_parts)
            else:
                # No isolated characters, use regular conversion
                pinyin_list = pinyin(word_text, style=Style.TONE)
                pinyin_text = ''.join([item[0] for item in pinyin_list])
            
            pinyin_text = pinyin_text.replace("(ér)", "(r)")
                
            row_texts[pinyin_col_index] = pinyin_text
            
        final_table.append(row_texts)
        firstrow = False

    # Visualization for debugging


    for row in final_table:
        print(row)


    return final_table


# Example usage
if __name__ == "__main__":
    img_path = 'image.png'
    final_table = process_image(img_path)
