<?php
header('Content-Type: application/json');

// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Check if file was uploaded
if (!isset($_FILES['image'])) {
    echo json_encode([
        'status' => 'error',
        'error' => 'No image file uploaded'
    ]);
    exit;
}

$file = $_FILES['image'];

// Check for upload errors
if ($file['error'] !== UPLOAD_ERR_OK) {
    echo json_encode([
        'status' => 'error',
        'error' => 'File upload failed: ' . $file['error']
    ]);
    exit;
}

// Check file type
$allowed_types = ['image/jpeg', 'image/png', 'image/jpg'];
if (!in_array($file['type'], $allowed_types)) {
    echo json_encode([
        'status' => 'error',
        'error' => 'Invalid file type. Only JPG and PNG files are allowed.'
    ]);
    exit;
}

// API endpoint
$api_url = 'https://riceapi-4n6n.onrender.com/predict';

// Create cURL request
$curl = curl_init();

// Create file upload data
$post_data = [
    'file' => new CURLFile($file['tmp_name'], $file['type'], $file['name'])
];

// Set cURL options
curl_setopt_array($curl, [
    CURLOPT_URL => $api_url,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $post_data,
    CURLOPT_HTTPHEADER => [
        'Accept: application/json'
    ],
    CURLOPT_TIMEOUT => 30, // 30 second timeout
    CURLOPT_CONNECTTIMEOUT => 10 // 10 second connection timeout
]);

// Execute the request
$response = curl_exec($curl);
$http_code = curl_getinfo($curl, CURLINFO_HTTP_CODE);
$error = curl_error($curl);

// Close cURL connection
curl_close($curl);

// Check for cURL errors
if ($error) {
    echo json_encode([
        'status' => 'error',
        'error' => 'API request failed: ' . $error
    ]);
    exit;
}

// Check HTTP response code
if ($http_code !== 200) {
    echo json_encode([
        'status' => 'error',
        'error' => 'API returned error code: ' . $http_code . '. Response: ' . $response
    ]);
    exit;
}

// Try to decode the response to check if it's valid JSON
$decoded_response = json_decode($response, true);
if (json_last_error() !== JSON_ERROR_NONE) {
    echo json_encode([
        'status' => 'error',
        'error' => 'Invalid JSON response from API: ' . $response
    ]);
    exit;
}

// Return the API response
echo $response;
?> 