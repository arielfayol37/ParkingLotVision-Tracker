import cv2

# Load the image
image = cv2.imread('test_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to identify black regions
_, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# Invert the binary image to identify black regions
binary = cv2.bitwise_not(binary)

# Find contours of the black regions
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw white rectangles around the black regions
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

# Display the result
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
