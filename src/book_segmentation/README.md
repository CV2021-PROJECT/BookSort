# Book Segmentation

assigned to 박신홍(@shp7724)

## How to Use?

```python
from book_segmentation import BookSpines

book_spines = BookSpines(row_images=row_images, verbose=False)
books = book_spines.get_books()
for book in books:
    rect, _ = book.rect()
    # do whatever you want ...
```
