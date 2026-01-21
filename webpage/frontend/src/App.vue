<script setup>
/**
 * DocSearch Application - PDF Viewer with Search and Highlight Functionality
 * =========================================================================
 * 
 * This application provides a specialized PDF viewing experience with the following features:
 * - Server-side PDF rendering to overcome browser PDF viewer limitations
 * - OCR text extraction with word coordinates stored in Elasticsearch
 * - Text search with precise highlighting on rendered PDF images
 * - Zoom and pan functionality for document navigation
 * - Dynamic DPI control for crisp rendering at all zoom levels
 * - Autocomplete suggestions for search terms
 */

  import { ref, watch, onMounted, onUnmounted, computed } from 'vue';
  import axios from 'axios';
  import {
    fetchSuggestions,
    handleBlur,
    selectSuggestion,
    suggestController,
    debounceTimer,
    ignoreNextFetch,
    suggestions,
    showSuggestions,
    searchTerm,
    hideSuggestions
  } from './components/Suggestions.js';

  const currentbuttonclicked = ref('')
  const myCheckbox = ref(false);

  // Index selection state
  const availableIndices = ref([]);
  const selectedIndex = ref(null);  // null = show landing page
  const indicesLoading = ref(true);

  // Fetch available indices on mount
  const fetchIndices = async () => {
    try {
      const resp = await axios.get('http://localhost:3000/indices');
      availableIndices.value = resp.data.indices || [];
    } catch (e) {
      console.error('Failed to fetch indices:', e);
    } finally {
      indicesLoading.value = false;
    }
  };

  const selectIndex = (indexName) => {
    selectedIndex.value = indexName;
  };

  const backToIndices = () => {
    selectedIndex.value = null;
    resultImage.value = null;
    allMatches.value = [];
    message.value = 'Click button to search...';
  };

  // Application constants
  const DEBUG_HIGHLIGHTS = true; // Set to true to show debug information on highlights

  /* const searchTerm = ref('');         // Current text in the search input field */
  const lastSearchedTerm = ref('');   // Stores the last successfully searched term (prevents UI updating during typing)
  const results = ref(null);          // Holds search results from the backend including match coordinates
  const message = ref('Click button to search...'); // Status message shown to the user

  // Toggle for highlighting all OCR words
  const highlightAllWords = ref(false);

  // Flag to prevent ResizeObserver from redrawing when we programmatically resize the image
  // This prevents double-drawing and zoom loops
  const isResizingProgrammatically = ref(false);
  
  // Multiple results handling
  const allMatches = ref([]);         // Array of all matching pages/files from the search
  const selectedMatchIndex = ref(0);  // Index of currently selected match from allMatches
  
  // Folder filtering
  const selectedFolderFilter = ref('all');  // 'all', 'Alyth 400 kV', or 'Tealing 275 kV'
  // Add computed property for unique folders
  const uniqueFolders = computed(() => {
    const folders = new Set();
    allMatches.value.forEach(match => {
      if (match.sourcefolder) {
        folders.add(match.sourcefolder);
      }
    });
    return Array.from(folders).sort();
  });

  // PDF interaction variables
  const resultImage = ref(null);  // URL path to the server-rendered page image
  const pageImage = ref(null);  // Reference to the image DOM element (populated via Vue ref attribute)
  const overlayCanvas = ref(null);  // Reference to the canvas overlay DOM element for highlighting
  const imageContainer = ref(null); // Reference to the container for zoom/pan functionality
  
  // Original/natural image dimensions (important for coordinate scaling)
  const imageWidth = ref(0);
  const imageHeight = ref(0);
  
  // Loading state indicator
  const isLoading = ref(false);
  
  // Zoom control variables
  const zoomLevel = ref(1);           // Current zoom level (1 = 100%, original size)
  const minZoom = 0.5;                // Minimum zoom level (50% of original size)
  const maxZoom = 10;                  // Maximum zoom level (500% of original size)
  const wheelZoomStep = 0.1;          // 10% zoom change per wheel tick
  const panPosition = ref({ x: 0, y: 0 }); // Current pan/scroll position (x,y coordinates)
  const isDragging = ref(false);      // Whether the user is currently panning (dragging the image)
  const dragStart = ref({ x: 0, y: 0 }); // Start position for drag operation (used to calculate movement)
  const mousePosition = ref({ x: 0, y: 0 }); //debugonly

  // Keep track of the current page being viewed
  const currentIndexName = ref(null);
  const currentPageNumber = ref(null);
  
  const onImageLoad = async () => {
    /* 
    Gets called after the image URL is set (resulImage) and the image finishes loading from the server.
    Applies pan and zoom and sets mouse listening events to track mouse position,
    usefull for zoom on mouseposition.
    */
    try {
      // Image is loaded, make sure it's visible
      isLoading.value = false;

      const img = pageImage.value;
      const canvas = overlayCanvas.value;
      if (!img || !canvas) {
        return;
      }
      
      // Store image dimensions for proper coordinate scaling
      imageWidth.value = img.naturalWidth;
      imageHeight.value = img.naturalHeight;

      // IMPORTANT: Size the image properly based on current zoom level
      // This will internally call applyZoomAndPan() that wiil call 
      // drawHighlights() so we don't need to call it again
      resetZoom();

      const canvasContainer = imageContainer.value;
      if (canvasContainer) {
        // Add mouse position tracking for canvas debugging
        const trackMousePosition = (event) => {
          const canvas = overlayCanvas.value;
          if (canvas) {
            const rect = canvas.getBoundingClientRect();
            mousePosition.value = {
              x: Math.round(event.clientX - rect.left),
              y: Math.round(event.clientY - rect.top)
            };
          }
        };

        canvasContainer.addEventListener('mousemove', trackMousePosition);
      }
    } catch (error) {
      console.error("Error in onImageLoad:", error);
      isLoading.value = false;
    }
  };

  // Force canvas to match current image display size
  // Needed when the zoom is made and to verify the match of dpi rendering
    const updateCanvasDimensions = () => {
      const img = pageImage.value;
      const canvas = overlayCanvas.value;

      if (img && canvas) {
        // Get rendered size of the image (considering CSS scaling)
        const displayWidth = img.clientWidth;
        const displayHeight = img.clientHeight;
        
        // Original image dimensions (from server)
        const originalWidth = img.naturalWidth;
        const originalHeight = img.naturalHeight;
        
        if (displayWidth > 0 && displayHeight > 0) {
          // Update canvas to match the image's current display size (use devicePixelRatio)
          const dpr = window.devicePixelRatio || 1; // Ensures webpage crispiness for every device pixel ratio
          canvas.style.width = `${displayWidth}px`;
          canvas.style.height = `${displayHeight}px`;
          canvas.width = Math.round(displayWidth * dpr);
          canvas.height = Math.round(displayHeight * dpr);
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, displayWidth, displayHeight);
          }
          
          // Store original image dimensions for coordinate scaling
          if (originalWidth > 0 && originalHeight > 0) {
            imageWidth.value = originalWidth;
            imageHeight.value = originalHeight;
          } else {
            imageWidth.value = displayWidth;
            imageHeight.value = displayHeight;
          }
          
          // Debug info for coordinate scaling
          if (DEBUG_HIGHLIGHTS) {
            console.log(`Canvas size: ${canvas.width}x${canvas.height}, Image size: ${imageWidth.value}x${imageHeight.value}, Scale: ${(canvas.width/imageWidth.value).toFixed(3)}x${(canvas.height/imageHeight.value).toFixed(3)}`);
          }
          
          return true;
        }
      }
      return false;
    };
  
  const drawHighlights = (matches) => {
    /* 
    Function that draws Highlights, there is no need to add retry logic,
    everytime a zoom event is called, this function is called 
    */
    if (!matches || matches.length === 0) {
      console.log("No matches to highlight");
      return;
    }

    const canvas = overlayCanvas.value;
    if (!canvas) {
      console.warn("Canvas reference missing when drawing highlights");
      return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    
    if(!updateCanvasDimensions()){
      return;
    }
    // Clear any existing highlights
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (imageWidth.value <= 0 || imageHeight.value <= 0) {
      return;
    }
    
    // Use CSS pixel dimensions for coordinate scaling, not canvas buffer size
    const img = pageImage.value;
    const cssWidth = img ? img.clientWidth : canvas.width;
    const cssHeight = img ? img.clientHeight : canvas.height;
    
    // Scale from image natural dimensions to CSS display dimensions
    const scaleX = (cssWidth > 0 && imageWidth.value > 0) ? 
                  (cssWidth / imageWidth.value) : 1;
    const scaleY = (cssHeight > 0 && imageHeight.value > 0) ? 
                  (cssHeight / imageHeight.value) : 1;
    
    // Draw each match as a highlighted rectangle
    matches.forEach((match, index) => {
        if (!match.coordinates || 
              typeof match.coordinates.x0 !== 'number' || 
              typeof match.coordinates.y0 !== 'number' || 
              typeof match.coordinates.x1 !== 'number' || 
              typeof match.coordinates.y1 !== 'number') {
            return;
        }
        
        try {
          const confidence = match.score || 0.5;
          const matchWord = match.word;
          if(!matchWord){
            return;
          }

          let x0 = match.coordinates.x0;
          let y0 = match.coordinates.y0;
          let x1 = match.coordinates.x1;
          let y1 = match.coordinates.y1;
          
          const appearsNormalized = x0 >= 0 && x0 <= 1 && y0 >= 0 && y0 <= 1 && 
                                  x1 >= 0 && x1 <= 1 && y1 >= 0 && y1 <= 1;
          
          if (appearsNormalized) {
            x0 *= imageWidth.value;
            y0 *= imageHeight.value;
            x1 *= imageWidth.value;
            y1 *= imageHeight.value;
          }
          
          // Apply scaling to CSS dimensions (not canvas buffer dimensions)
          const x = x0 * scaleX;
          const y = y0 * scaleY;
          const width = Math.max(8, (x1 - x0) * scaleX);
          const height = Math.max(8, (y1 - y0) * scaleY);
          
          if (width <= 0 || height <= 0 || isNaN(x) || isNaN(y) || isNaN(width) || isNaN(height)) {
            return;
          }
          
          let finalX = x;
          let finalY = y;
          let finalWidth = width;
          let finalHeight = height;
          
          const minHighlightWidth = 8;
          const minHighlightHeight = 10;
          
          if (finalWidth < minHighlightWidth) {
            finalX -= (minHighlightWidth - finalWidth) / 2;
            finalWidth = minHighlightWidth;
          }
          
          if (finalHeight < minHighlightHeight) {
            finalY -= (minHighlightHeight - finalHeight) / 2;
            finalHeight = minHighlightHeight;
          }
          
          // Draw in CSS pixel coordinates (canvas context is already scaled by DPR via setTransform)
          ctx.fillStyle = 'rgba(0, 255, 255, 0.4)';
          ctx.fillRect(finalX, finalY, finalWidth, finalHeight);
          
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.lineWidth = 1;
          ctx.strokeRect(finalX, finalY, finalWidth, finalHeight);

          if (DEBUG_HIGHLIGHTS) {
            const debugInfo = `"${matchWord}" CSS:[${finalX.toFixed(0)},${finalY.toFixed(0)}] ` +
                            `Size:${finalWidth.toFixed(0)}x${finalHeight.toFixed(0)} ` +
                            `Conf:${(confidence * 100).toFixed(0)}%`;
                            
            const fontSize = Math.max(10, Math.min(12, zoomLevel.value * 8));
            ctx.font = `${fontSize}px monospace`;
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(finalX, finalY - fontSize - 4, ctx.measureText(debugInfo).width + 6, fontSize + 6);
            ctx.fillStyle = '#00ff00';
            ctx.fillText(debugInfo, finalX + 3, finalY - 3);
        }
      } catch (err) {
        console.error(`Error drawing highlight for match #${index}:`, err);
      }
    });
  };
  
  // Track resize observer reference
  let resizeObserver = null;
  
  // Handle window resize to adjust canvas size
  const handleResize = () => {
    const img = pageImage.value;
    const canvas = overlayCanvas.value;
    
    if (!img || !canvas) return;
    
    // Get the currently selected match
    const matchData = allMatches.value[selectedMatchIndex.value];
    if (!matchData || !matchData.matches) return;
    
    // Update canvas dimensions
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    // Redraw highlights with new dimensions
    drawHighlights(matchData.matches);
  };
  
  /**
   * onMounted lifecycle hook - Sets up event listeners and observers
   * 
   * This function initializes all the event handling infrastructure needed for:
   * 1. Window resizing - Ensures canvas stays aligned with image
   * 2. Mouse/touch drag interactions - Enables the pan functionality
   * 3. Element size monitoring - Handles layout changes via ResizeObserver
   * 
   * Each event listener is carefully chosen to ensure cross-browser and
   * cross-device compatibility, including desktop mouse and mobile touch support.
   */
  onMounted(() => {
    fetchIndices();
    // Add global mouse event listeners for pan functionality
    // Using global listeners ensures smooth dragging even when mouse moves outside element
    window.addEventListener('mousemove', doDrag);
    window.addEventListener('mouseup', stopDrag);
    
    // Add touch event listeners for mobile support
    // These provide equivalent functionality to mouse events on touch devices
    // passive:false allows us to use preventDefault to prevent scrolling while dragging
    window.addEventListener('touchmove', doDrag, { passive: false });
    window.addEventListener('touchend', stopDrag);
    window.addEventListener('touchcancel', stopDrag);
    

    // Add mouse position tracking for canvas debugging DEBUG ONLY
    const trackMousePosition = (event) => {
    const canvas = overlayCanvas.value;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;  // Add this
      
      mousePosition.value = {
        x: Math.round((event.clientX - rect.left) * dpr),  // Scale by DPR
        y: Math.round((event.clientY - rect.top) * dpr)    // Scale by DPR
      };
    }
  };

    // Track mouse position on canvas container
    const canvasContainer = imageContainer.value;
    if (canvasContainer) {
      canvasContainer.addEventListener('mousemove', trackMousePosition);
    }


    // Create a DPR-aware ResizeObserver to monitor image size changes.
    // This implementation uses entry.contentRect, scales the canvas internal buffer
    // by devicePixelRatio, and debounces redraws to avoid thrashing during rapid layout changes.
    if (window.ResizeObserver) {
      // Clean up any existing observer before creating a new one
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      
      // Debounced redraw helper (scope-local)
      const debouncedRedraw = (() => {
        let timer = null;
        const wait = 80; // ms
        return (matches) => {
          if (timer) clearTimeout(timer);
          timer = setTimeout(() => {
            try {
              drawHighlights(matches);
            } catch (e) {
              console.error('Debounced redraw failed:', e);
            }
          }, wait);
        };
      })();

      resizeObserver = new ResizeObserver(entries => {
        // Skip if we're programmatically resizing (prevents double-draw and loops)
        if (isResizingProgrammatically.value) {
          return;
        }
        
        for (const entry of entries) {
          // Only handle the entry for our tracked image element
          if (entry.target !== pageImage.value) continue;

          const img = pageImage.value;
          const canvas = overlayCanvas.value;
          if (!img || !canvas) return;

          // Use contentRect for measured layout size (best compatibility)
          const rect = entry.contentRect || { width: img.clientWidth, height: img.clientHeight };
          const displayW = Math.max(1, Math.round(rect.width));
          const displayH = Math.max(1, Math.round(rect.height));

          // Device pixel ratio for crisp rendering on HiDPI screens
          const dpr = window.devicePixelRatio || 1;

          // Set CSS size so the canvas occupies the expected layout box
          canvas.style.width = `${displayW}px`;
          canvas.style.height = `${displayH}px`;

          // Set internal pixel buffer larger for high DPI then scale the context
          canvas.width = Math.round(displayW * dpr);
          canvas.height = Math.round(displayH * dpr);

          const ctx = canvas.getContext('2d');
          if (ctx) {
            // Map drawing operations from CSS pixels to device pixels
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            // Clear in CSS pixels (transform maps to correct internal pixels)
            ctx.clearRect(0, 0, displayW, displayH);
          }

          // Update stored natural sizes if available (used by drawHighlights scaling)
          if (img.naturalWidth > 0 && img.naturalHeight > 0) {
            imageWidth.value = img.naturalWidth;
            imageHeight.value = img.naturalHeight;
          }

          // Determine current matches to redraw
          const matchData = allMatches.value[selectedMatchIndex.value]; 
          if (matchData?.matches) {
            console.log('ResizeObserver detected external resize - redrawing highlights');
            debouncedRedraw(matchData.matches); // Always use current match data
          }

          break; // handled our image
        }
      });
    }
    else{
      print("Error while window resizing, window.ResizeObserver")
    }
  });
  
  /**
   * Watch handler for pageImage - Sets up ResizeObserver when image changes
   * 
   * This watcher ensures that any time the image reference changes (such as when loading
   * a new page), we properly set up the ResizeObserver to monitor the new image element.
   * 
   * The observer is critical for keeping the highlight canvas properly sized and
   * positioned over the image, especially during layout changes or when loading new content.
   */
  watch(pageImage, (newImg, oldImg) => {
    // If a previous image was observed, unobserve it to avoid duplicate callbacks
    if (oldImg && resizeObserver) {
      try {
        resizeObserver.unobserve(oldImg);
        console.log('Unobserved previous image from ResizeObserver');
      } catch (e) {
        // ignore
      }
    }

    // If we have a new image and it's valid, observe it
    if (newImg && resizeObserver) {
      console.log("Setting up ResizeObserver on new image");
      try {
        resizeObserver.observe(newImg);
      } catch (e) {
        console.error('Failed to observe new image:', e);
      }
    }
  });
  
  // Zoom in function - progressive resolution loading
  const zoomIn = () => {
    if (zoomLevel.value < maxZoom) {
      // Calculate new zoom based on step
      // For smooth zooming: smaller steps near 1.0, larger steps at higher zoom
      const step = zoomLevel.value < 2 ? 0.25 : (zoomLevel.value < 4 ? 0.5 : 1.0);
      zoomLevel.value = Math.min(maxZoom, zoomLevel.value + step);
      
      // Apply the zoom immediately (uses current image)
      applyZoomAndPan();
    }
  };

  // Zoom out function - decreases zoom level
  const zoomOut = () => {
    if (zoomLevel.value > minZoom) {
      const step = zoomLevel.value <= 2 ? 0.25 : (zoomLevel.value <= 4 ? 0.5 : 1.0);
      zoomLevel.value = Math.max(minZoom, zoomLevel.value - step);
      
      applyZoomAndPan();
    }
  };
  
  // Reset zoom and pan to defaults
  const resetZoom = () => {
    zoomLevel.value = 1;
    panPosition.value = { x: 0, y: 0 };
    applyZoomAndPan();
  };
  
  // Apply the current zoom level and pan position to the image and canvas
  const applyZoomAndPan = () => {
    const container = imageContainer.value;
    const img = pageImage.value;
    if (!container || !img) return;

    // Set flag to prevent ResizeObserver from triggering during our programmatic resize
    isResizingProgrammatically.value = true;

    // CRITICAL: Instead of using CSS scale() transform, we directly control the image size
    // This allows the browser to use the full resolution of the source image
    // avoiding pixelation when zooming in
    
    // Get the natural (full) dimensions of the image
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    
    if (naturalWidth === 0 || naturalHeight === 0) {
      // Image not loaded yet
      isResizingProgrammatically.value = false;
      return;
    }
    
    // Calculate viewport dimensions to fit the image initially
    const viewport = container.parentElement;
    if (!viewport) {
      isResizingProgrammatically.value = false;
      return;
    }
    
    const viewportWidth = viewport.clientWidth;
    const viewportHeight = viewport.clientHeight;
    
    // Calculate the scale needed to fit the image in the viewport initially (at 100% zoom)
    const scaleToFit = Math.min(
      viewportWidth / naturalWidth,
      viewportHeight / naturalHeight,
      1.0  // Never scale UP beyond natural size at 100% zoom
    );
    
    // Calculate the actual display size based on zoom level
    // At 100% zoom (zoomLevel = 1.0), the image fits the viewport
    // At 200% zoom (zoomLevel = 2.0), the image is twice as large
    const displayWidth = naturalWidth * scaleToFit * zoomLevel.value;
    const displayHeight = naturalHeight * scaleToFit * zoomLevel.value;
    
    // Set the image size directly - this makes the browser use the full resolution
    // IMPORTANT: These must be applied synchronously before any other calculations
    img.style.width = `${displayWidth}px`;
    img.style.height = `${displayHeight}px`;
    
    // Apply only translation for panning (no scale transform)
    // Also applied synchronously
    container.style.transform = `translate(${panPosition.value.x}px, ${panPosition.value.y}px)`;
    container.style.transformOrigin = 'top left';
    
    // Redraw highlights and clear flag asynchronously (non-critical for zoom positioning)
    requestAnimationFrame(() => {
      // After applying zoom, redraw the highlights
      const currentMatchData = allMatches.value[selectedMatchIndex.value];
      if (currentMatchData && currentMatchData.matches) {
        drawHighlights(currentMatchData.matches);
      }
      
      // Clear the flag after a brief delay to allow ResizeObserver to process
      setTimeout(() => {
        isResizingProgrammatically.value = false;
      }, 100);
    });
  };

  // Start dragging (panning) the image
  const startDrag = (event) => {
    // Allow dragging at any zoom level - removed restriction
    
    // Handle both mouse and touch events
    const isTouchEvent = event.type === 'touchstart';
    
    // For mouse events, only start drag on primary mouse button (left click)
    if (!isTouchEvent && event.button !== 0) return;
    
    // Prevent default to avoid selecting text or dragging the image as a draggable element
    event.preventDefault();
    
    isDragging.value = true;
    
    // Get client coordinates from either mouse or touch event
    const clientX = isTouchEvent ? event.touches[0].clientX : event.clientX;
    const clientY = isTouchEvent ? event.touches[0].clientY : event.clientY;
    
    dragStart.value = {
      x: clientX,
      y: clientY,
      panX: panPosition.value.x,
      panY: panPosition.value.y
    };
    
    // Change cursor to indicate dragging
    if (imageContainer.value) {
      imageContainer.value.style.cursor = 'grabbing';
    }
  };
  
  // Continue dragging (panning) the image
  const doDrag = (event) => {
    if (!isDragging.value) return;
    
    // Handle both mouse and touch events
    const isTouchEvent = event.type === 'touchmove';
    
    // Prevent default during drag
    if (event.cancelable) {
      event.preventDefault();
    }
    
    // Get client coordinates from either mouse or touch event
    const clientX = isTouchEvent ? event.touches[0].clientX : event.clientX;
    const clientY = isTouchEvent ? event.touches[0].clientY : event.clientY;
    
    // Calculate the new position based on the mouse movement
    const dx = clientX - dragStart.value.x;
    const dy = clientY - dragStart.value.y;
    
    const newX = dragStart.value.panX + dx;
    const newY = dragStart.value.panY + dy;

    panPosition.value = {
      x: newX,
      y: newY
    };
    
    // Apply the new pan position to the container
    // IMPORTANT: Only apply translate, NOT scale - the image size is controlled directly
    if (imageContainer.value) {
      imageContainer.value.style.transform = `translate(${panPosition.value.x}px, ${panPosition.value.y}px)`;
    }
  };
  
  // Stop dragging (panning) the image
  const stopDrag = () => {
    if (!isDragging.value) return;
    
    isDragging.value = false;
    
    // Reset cursor
    if (imageContainer.value) {
      imageContainer.value.style.cursor = 'grab';
    }
  };
  
  // Calculate zoom centered on mouse position
  const zoomAtPosition = (newZoom, mouseX, mouseY) => {
    if (!imageContainer.value || !pageImage.value) return;
    
    const container = imageContainer.value;
    const img = pageImage.value;
    const oldZoom = zoomLevel.value;
    
    // Get the current display dimensions before zoom change
    const currentWidth = img.clientWidth;
    const currentHeight = img.clientHeight;
    
    if (currentWidth === 0 || currentHeight === 0) {
      // Image not yet sized, just update zoom level
      zoomLevel.value = newZoom;
      applyZoomAndPan();
      return;
    }
    
    // Get the viewport (parent of container)
    const viewport = container.parentElement;
    if (!viewport) {
      zoomLevel.value = newZoom;
      applyZoomAndPan();
      return;
    }
    
    // Calculate mouse position relative to the viewport
    const viewportRect = viewport.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const mouseRelativeToViewportX = mouseX - viewportRect.left;
    const mouseRelativeToViewportY = mouseY - viewportRect.top;
    const mouseRelativeToContainerX = mouseX - containerRect.left;
    const mouseRelativeToContainerY = mouseY - containerRect.top;

    if (oldZoom <= 0) {
      return;
    }
    
    // Calculate the point on the BASE image (100% zoom equivalent) under the mouse
    // The current display shows the image at: baseSize * oldZoom
    // The distance from pan origin to mouse in current display: (mousePos - pan)
    // To get the coordinate on the base image, we divide by oldZoom:
    const baseImagePointX = mouseRelativeToContainerX / oldZoom;
    const baseImagePointY = mouseRelativeToContainerY / oldZoom;
    
    // Update zoom level
    zoomLevel.value = newZoom;
    
    // Calculate the new pan position so that the same base image point stays under the mouse
    // At the new zoom level, that base point will be at: baseImagePoint * newZoom
    // So we need: pan = mousePos - (baseImagePoint * newZoom)
    const newPanX = mouseRelativeToViewportX - (baseImagePointX * newZoom);
    const newPanY = mouseRelativeToViewportY - (baseImagePointY * newZoom);
    
    // Update pan position
    panPosition.value = { x: newPanX, y: newPanY };
    
    // Apply the new transform
    applyZoomAndPan();
  };
  
  // Handle mouse wheel for zooming
  const handleWheel = (event) => {
    if (!imageContainer.value) return;
    // Prevent default to avoid page scrolling
    event.preventDefault();
    
    const zoomMultiplier = 1 + wheelZoomStep;
    const scrollingOut = event.deltaY > 0;

    const targetZoom = scrollingOut
      ? zoomLevel.value / zoomMultiplier
      : zoomLevel.value * zoomMultiplier;

    const newZoom = Math.max(minZoom, Math.min(maxZoom, targetZoom));

    if (Math.abs(newZoom - zoomLevel.value) < 1e-4) {
      return;
    }
    
    // Apply zoom centered at mouse position
    zoomAtPosition(newZoom, event.clientX, event.clientY);
  };
  
  // Clean up event listeners when component is unmounted
  onUnmounted(() => {
    // Remove all window event listeners for resize and mouse/touch events
    window.removeEventListener('resize', handleResize);
    window.removeEventListener('mousemove', doDrag);
    window.removeEventListener('mouseup', stopDrag);
    window.removeEventListener('touchmove', doDrag, { passive: false });
    window.removeEventListener('touchend', stopDrag);
    window.removeEventListener('touchcancel', stopDrag);
    
    // Clean up ResizeObserver
    // disconnect() stops all observations and frees up resources
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
  });
  
  watch(searchTerm, fetchSuggestions);

  // Computed property to filter matches by selected folder with cached indices
  const filteredMatches = computed(() => {
    if (selectedFolderFilter.value === 'all') {
      // Add original index to each match for efficient lookups
      return allMatches.value.map((match, index) => ({ ...match, _originalIndex: index }));
    }
    // Filter and cache original indices
    return allMatches.value
      .map((match, index) => ({ ...match, _originalIndex: index }))
      .filter(match => match.sourcefolder === selectedFolderFilter.value);
  });
  
  // Calculate average confidence score for a match
  const getAverageConfidence = (match) => {
    if (!match || !match.matches || match.matches.length === 0) return 0.5;
    
    // Extract scores from all matches, defaulting to 0.5 if not present
    const scores = match.matches.map(m => m.score || 0.5);
    
    // Calculate average
    const sum = scores.reduce((total, score) => total + score, 0);
    return sum / scores.length;
  };
  
  // Function to load and display a specific search result
  const displayResult = async (matchItem, index = null) => {
    // Update selected index if provided
    if (index !== null) {
      selectedMatchIndex.value = index;
    }

    // Set loading state
    isLoading.value = true;
    
    currentbuttonclicked.value = matchItem.fileName;  

    // Get the folder, file name and page number for this result
    const folderName = matchItem.folderName || '';
    const fileName = matchItem.fileName || "";
    const pageNumber = matchItem.pageNumber || 1;
    
    // Check if we're viewing the same page as before
    const isSamePage = (fileName === currentIndexName.value && 
                        pageNumber === currentPageNumber.value);
    
    if (isSamePage && !isLoading.value) {
      // If it's the same page, don't reload the image, just redraw highlights
      drawHighlights(matchItem.matches);
    } else {
      // Different page or first load
      
      // Remember the current page being viewed
      currentIndexName.value = fileName;
      currentPageNumber.value = pageNumber;
      
      // Build image URL - use pre-rendered images (200 DPI from OCR indexing)
      const renderUrl = `http://localhost:3000/renderedpages/${fileName}_page_${pageNumber}.png`;
      
      // Hold the ResizeObserver guard until the zoom sizing finishes
      isResizingProgrammatically.value = true;

      // Set image URL directly - pre-rendered images are always available
      resultImage.value = renderUrl;
      
      // Simple fallback check
      fetch(renderUrl, { method: 'HEAD' })
        .then(response => {
          if (response.ok) {
            isLoading.value = false;
          } else {
            isLoading.value = false;
            message.value = "Pre-rendered image not available.";
            isResizingProgrammatically.value = false;
          }
        })
        .catch(error => {
          console.error("Error checking pre-rendered image:", error);
          isLoading.value = false;
          message.value = "Error fetching pre-rendered image.";
          isResizingProgrammatically.value = false;
        });
    }
  };

  const SearchButtonClicked = async () => {
    if (!searchTerm.value.trim()) {
      message.value = 'Please enter a search term';
      return;
    }
    
    // Clear the current image if one is displayed
    resultImage.value = null;
    
    message.value = 'Searching...';
    
    // Store the current search term when search is executed
    lastSearchedTerm.value = searchTerm.value.trim();
    
    try {
      const response = await axios.post('http://localhost:3000/wordcheck', {
        searchTerm: lastSearchedTerm.value,
        highlightAllWords: highlightAllWords.value,
        exactMatch: myCheckbox.value,
        indexName: selectedIndex.value || '*'
      });
      
      // Handle JSON response properly
      if (response.data.success) {
        // Store original results for backward compatibility
        results.value = response.data.results;
        message.value = response.data.message;

        // Store all matching pages/files
        allMatches.value = response.data.results.allMatches || [];
        
        console.log(`Found ${allMatches.value.length} matching pages`);
        console.log("Search result data:", response.data.results);
        
        // Reset selected index but don't display any result yet
        selectedMatchIndex.value = -1;
        
        // No automatic display of results
        // User must click a result button to view the image
        isLoading.value = false;
      } else {
        // Search failed
        results.value = null;
        allMatches.value = [];
        message.value = response.data.message;
        isLoading.value = false;
      }
      
    } catch (error) {
      if (error.response && error.response.data) {
        message.value = error.response.data.error || error.response.data.message;
      } else {
        message.value = "Connection error: " + error.message;
      }
      isLoading.value = false;
    }    
  };

  // Handle toggle change to re-search if results exist
  const onHighlightModeChange = () => {
    if (allMatches.value.length > 0) {
      // Re-run search with new mode
      SearchButtonClicked();
    }
  };
</script>

<template>
  <!-- Main container -->
  <div class="main-container">
    <!-- Title bar -->
    <header class="title-bar">
      <h1>DocSearch</h1>
      <button v-if="selectedIndex" class="back-button" @click="backToIndices">← Back to Indices</button>
    </header>

    <!-- Landing page: Index selection -->
    <div v-if="!selectedIndex" class="landing-container">
      <h2>Select an Index</h2>
      <div v-if="indicesLoading" class="loading">Loading indices...</div>
      <div v-else-if="availableIndices.length === 0" class="no-indices">No indices found</div>
      <div v-else class="indices-grid">
        <button
          v-for="idx in availableIndices"
          :key="idx.name"
          class="index-card"
          @click="selectIndex(idx.name)"
        >
          <span class="index-name">{{ idx.name }}</span>
          <span class="index-docs">{{ idx.docs.toLocaleString() }} documents</span>
        </button>
      </div>
    </div>

    <!-- Search interface (existing) -->
    <div v-else class="content-container">
      <!-- Left panel for search interface -->
      <div class="left-panel">
        <!-- Word to search section -->
        <div class="section">
          <p class="section-title">Word to search:</p>
          
          <!-- Folder filter buttons -->
          <div v-if="uniqueFolders.length > 1" class="filter-buttons">
            <button 
              class="filter-button" 
              :class="{ 'active': selectedFolderFilter === 'all' }"
              @click="selectedFolderFilter = 'all'"
            >
              All
            </button>
            <button 
              v-for="folder in uniqueFolders"
              :key="folder"
              class="filter-button" 
              :class="{ 'active': selectedFolderFilter === folder }"
              @click="selectedFolderFilter = folder"
            >
              {{ folder }}
            </button>
          </div>
          <div class="search-wrapper">
            <input 
              type="search" 
              v-model="searchTerm" 
              placeholder="Search text..."
              @keyup.enter="SearchButtonClicked"
              @keydown.enter="hideSuggestions"
              @blur="handleBlur"
              @focus="showSuggestions = suggestions.length > 0"
            >
            <!-- Suggestions dropdown -->
            <div v-if="showSuggestions && suggestions.length" class="suggestions-dropdown">
              <div 
                v-for="(s, i) in suggestions" 
                :key="i" 
                class="suggestion-item"
                @mousedown="selectSuggestion(s, SearchButtonClicked)"
              >
                {{ s }}
              </div>
            </div>
            <div class="checkbox-container">
              <!-- Checkbox to manage search  -->
              <label class="custom-checkbox">
                <span class="label-text">Exact Match</span>
                <input type="checkbox" v-model="myCheckbox" />
                <span class="checkmark"></span>
              </label>
            </div>
          </div>
          
          <div v-if="DEBUG_HIGHLIGHTS" class="toggle-container">
            <label class="toggle-label">
              <input 
                type="checkbox" 
                v-model="highlightAllWords"
                @change="onHighlightModeChange"
              />
              <span class="toggle-text">Highlight all OCR words</span>
            </label>
          </div>

          <!-- Search button -->
          <button @click="SearchButtonClicked">Search</button>
        </div>
        
        <!-- Results section -->
        <div class="section">
          <p class="section-title">Results:</p>
          <p>Currently on: {{ currentbuttonclicked }}</p>
          <div class="results">
            <div v-if="!allMatches || allMatches.length === 0" 
                 class="message" :class="{ 'highlight-message': message === 'Click button to search...' }">
              {{ message }}
            </div>
            <div v-else-if="filteredMatches.length === 0" class="message">
              No results in selected folder
            </div>
            <div v-else>
              <!-- List of results as buttons -->
              <div class="results-list">
                <button 
                  v-for="(match, filteredIndex) in filteredMatches" 
                  :key="`${match.fileName}-${match.pageNumber}-${filteredIndex}`"
                  class="result-button"
                  :class="{ 'selected': match._originalIndex === selectedMatchIndex }"
                  @click="displayResult(match, match._originalIndex)"
                >
                  <!-- Add confidence indicator -->
                  <span 
                    class="confidence-indicator" 
                    :class="{
                      'high-confidence': getAverageConfidence(match) >= 0.7, 
                      'medium-confidence': getAverageConfidence(match) >= 0.4 && getAverageConfidence(match) < 0.7,
                      'low-confidence': getAverageConfidence(match) < 0.4
                    }"
                    :title="`Average confidence: ${Math.round(getAverageConfidence(match) * 100)}%`"
                  >
                    {{ Math.round(getAverageConfidence(match) * 100) }}%
                  </span>
                  <span class="folder-name">{{ match.sourcefolder }}</span> / 
                  <span class="file-name">{{ match.fileName }}.pdf</span> Page 
                  <span class="page-number">{{ match.pageNumber }}</span>
                  <span class="match-count">({{ match.matches.length }} matches)</span>
                  <span class="detected-words" v-if="match.matches && match.matches.length > 0">
                    - "{{ [...new Set(match.matches.map(m => m.word))].join('", "') }}"
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Right panel for image display -->
      <div class="right-panel">
        <!-- Image display with highlight overlay -->
        <div v-if="resultImage" class="image-container">
          <!-- Show loading message only while loading -->
          <div v-if="isLoading" class="pdf-loading">Loading image...</div>
            
            <!-- Container with relative positioning for stacked elements -->
            <div 
              class="page-viewport"
              @wheel.prevent="handleWheel"
            >
            <!-- Always show the container for the image when not loading -->
            <div v-show="!isLoading" class="rendered-page-container">
              <!-- Zoom controls -->
              <div class="zoom-controls">
                <button @click="zoomOut" class="zoom-button" title="Zoom Out">
                  <span>-</span>
                </button>
                <span class="zoom-level">{{ Math.round(zoomLevel * 100) }}%</span>
                <button @click="zoomIn" class="zoom-button" title="Zoom In">
                  <span>+</span>
                </button>
                <button @click="resetZoom" class="zoom-button" title="Reset Zoom">
                  <span>Reset</span>
                </button>
              </div>
                <div 
                ref="imageContainer" 
                class="page-overlay-container"
                :class="{ 'draggable': true, 'dragging': isDragging }"
                @mousedown.prevent="startDrag"
                @touchstart.prevent="startDrag"
              >
                <!-- The rendered page image -->
                <img 
                  ref="pageImage" 
                  :src="resultImage" 
                  class="rendered-page" 
                  @load="onImageLoad"
                  @error="(e) => { console.error('Image failed to load:', e); isLoading.value = false; }"
                  alt="Rendered PDF Page"
                  draggable="false"
                  crossorigin="anonymous"
                />
                <!-- Canvas overlay for highlighting words - positioned on top of the image -->
                <canvas 
                  ref="overlayCanvas" 
                  class="highlight-overlay"
                ></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
.mouse-position-label {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-family: monospace;
  z-index: 20;
  pointer-events: none;
}

/* Global styles to ensure full page coverage */
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  overflow: hidden;
  background-color: #1e1e1e;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Critical: This ensures the Vue app takes full width */
#app {
  height: 100vh;
  width: 100vw;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.rendered-page-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  height: 100%;
  overflow: auto;
}

.page-viewport {
  width: 100%;
  height: 100%;
  overflow: hidden;
  position: relative;
  display: block;
  margin-bottom: 10px;
}

.page-overlay-container {
  position: absolute;
  top: 0;
  left: 0;
  display: block;
  transform-origin: top left;
  will-change: transform; /* Performance optimization for transforms */
}

.page-overlay-container.draggable {
  cursor: grab;
}

.page-overlay-container.dragging {
  cursor: grabbing;
}

.rendered-page {
  /* CRITICAL: Do NOT set width/height here - they are controlled by JavaScript */
  /* This allows the image to be displayed at its natural resolution when zoomed */
  /* The applyZoomAndPan() function controls the actual size dynamically */
  
  display: block;        /* Remove extra space under the image */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  transform-origin: top left;
  
  /* Prevent flash of unsized image */
  min-width: 0;
  min-height: 0;
  
  /* Force smooth, high-quality image scaling */
  /* Use only smooth/high-quality rendering - no pixelated/crisp-edges */
  image-rendering: auto;                       /* Let browser decide (usually high quality) */
  image-rendering: -webkit-optimize-contrast;  /* Safari - uses high-quality scaling */
  image-rendering: smooth;                      /* Standard - bicubic/bilinear interpolation */
  -ms-interpolation-mode: bicubic;             /* IE/Edge fallback */
  
  /* Additional properties for maximum quality in Chromium browsers */
  image-rendering: high-quality;                /* Non-standard but supported in some browsers */
  image-rendering: -webkit-optimize-quality;    /* Webkit optimization for quality */
  
  /* Hardware acceleration for smooth rendering */
  transform: translateZ(0);
  -webkit-transform: translateZ(0);
  will-change: transform;
}

.highlight-overlay {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Allows clicking through to the image */
  will-change: transform; /* Performance optimization */
  transform-origin: top left;
  z-index: 10; /* Ensure canvas is above the image */
  opacity: 1; /* Make sure the canvas is fully visible */
}

.zoom-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 10px;
  gap: 5px;
  z-index: 5;
}

.zoom-button {
  background-color: #333;
  border: 1px solid #555;
  color: #fff;
  width: 32px;
  height: 32px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.zoom-button:last-child {
  width: auto;
  padding: 0 10px;
  font-size: 12px;
}

.zoom-button:hover {
  background-color: #444;
}

.zoom-level {
  min-width: 60px;
  text-align: center;
  padding: 5px;
  background-color: #333;
  border: 1px solid #555;
  border-radius: 4px;
  font-size: 14px;
}

/* DPI indicator removed */

.page-info {
  margin-top: 10px;
  color: #aaa;
  text-align: center;
}

.pdf-loading {
  color: #888;
  font-style: italic;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 1.2rem;
}

/* The hidden class is no longer needed as we're using v-show */
</style>

<style scoped>
.checkbox-container {
  margin: 10px 0;
}

.custom-checkbox {
  display: flex;
  align-items: center;
  cursor: pointer;
  user-select: none;
  position: relative;
}

/* Hide the default checkbox */
.custom-checkbox input[type="checkbox"] {
  position: absolute;
  opacity: 0;
  cursor: pointer;
  height: 0;
  width: 0;
}

/* Custom checkbox box */
.checkmark {
  position: relative;
  height: 20px;
  width: 20px;
  background-color: #333;
  border: 2px solid #555;
  border-radius: 4px;
  margin-right: 10px;
  transition: all 0.3s ease;
}

/* Hover effect */
.custom-checkbox:hover input ~ .checkmark {
  background-color: #3a3a3a;
  border-color: #0078d4;
}

/* When checked */
.custom-checkbox input:checked ~ .checkmark {
  background-color: #0078d4;
  border-color: #0078d4;
}

/* Checkmark icon (hidden by default) */
.checkmark:after {
  content: "";
  position: absolute;
  display: none;
  left: 6px;
  top: 2px;
  width: 5px;
  height: 10px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

/* Show checkmark when checked */
.custom-checkbox input:checked ~ .checkmark:after {
  display: block;
}

.label-text {
  margin-right: 10px;
  color: #cccccc;
  font-size: 14px;
}

/* Focus state for accessibility */
.custom-checkbox input:focus ~ .checkmark {
  outline: 2px solid #0078d4;
  outline-offset: 2px;
}

/* Reset and base styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* Main container styling - full screen with dark theme */
.main-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  background-color: #1e1e1e;
  color: #e0e0e0;
  overflow: hidden;
}

/* Title bar styling */
.title-bar {
  background-color: #2d2d2d;
  border-bottom: 1px solid #000;
  padding: 8px 0;
  text-align: center;
  width: 100%;
  position: relative;
}

.title-bar h1 {
  margin: 0;
  font-size: 18px;
  font-weight: bold;
  color: #ffffff;
  display: inline;
}

.back-button {
  position: absolute;
  left: 20px;
  top: 50%;
  transform: translateY(-50%);
  background: #0078d4;
  border: none;
  color: white;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.back-button:hover {
  background: #0086f0;
}

/* Landing page styles */
.landing-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.landing-container h2 {
  color: #fff;
  margin-bottom: 30px;
  font-size: 24px;
}

.loading, .no-indices {
  color: #888;
  font-size: 16px;
}

.indices-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  max-width: 1000px;
  width: 100%;
}

.index-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 30px 20px;
  background: #2a2a2a;
  border: 2px solid #444;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.index-card:hover {
  background: #333;
  border-color: #0078d4;
  transform: translateY(-2px);
}

.index-name {
  font-size: 18px;
  font-weight: bold;
  color: #0078d4;
  margin-bottom: 8px;
}

.index-docs {
  font-size: 14px;
  color: #888;
}

/* Two-column layout container - fills all available space */
.content-container {
  display: flex;
  flex: 1;
  height: calc(100vh - 38px); /* Account for smaller title bar height */
  width: 100%;
  overflow: hidden;
}

/* Left panel styling */
.left-panel {
  width: 30%;
  min-width: 250px;
  padding: 20px;
  border-right: 1px solid #000;
  background-color: #252525;
  overflow-y: auto;
  max-height: 100%;
}

/* Right panel styling */
.right-panel {
  width: 70%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #252525;
  padding: 20px;
  overflow: hidden;
}

/* Section styling */
.section {
  margin-bottom: 20px;
  width: 100%;
  display: flex;
  flex-direction: column;
  flex: 1;
}

.section-title {
  font-weight: bold;
  margin-bottom: 5px;
  color: #cccccc;
}

.section-content {
  color: #0078d4;
}

/* Search wrapper and input styling */
.search-wrapper {
  position: relative;
  margin-bottom: 10px;
  width: 100%;
}

input {
  width: 100%;
  padding: 8px;
  border: 1px solid #555;
  border-radius: 4px;
  background-color: #333;
  color: #fff;
}

input::placeholder {
  color: #888;
}

button {
  padding: 8px 16px;
  background-color: #0078d4;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #0086f0;
}

/* Filter buttons styling */
.filter-buttons {
  display: flex;
  gap: 6px;
  margin-bottom: 12px;
  width: 100%;
}

.filter-button {
  flex: 1;
  padding: 6px 10px;
  font-size: 12px;
  background-color: #2a2a2a;
  color: #ccc;
  border: 1px solid #444;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.filter-button:hover {
  background-color: #3a3a3a;
  border-color: #555;
}

.filter-button.active {
  background-color: #0078d4;
  color: white;
  border-color: #0086f0;
}

/* Toggle switch styling */
.toggle-container {
  margin: 10px 0;
}

.toggle-label {
  display: flex;
  align-items: center;
  cursor: pointer;
  user-select: none;
}

.toggle-label input[type="checkbox"] {
  width: auto;
  margin-right: 8px;
  cursor: pointer;
}

.toggle-text {
  color: #cccccc;
  font-size: 14px;
}

/* Suggestions dropdown styling */
.suggestions-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  width: 100%;
  max-height: 200px;
  overflow-y: auto;
  background: #333;
  border-radius: 0 0 4px 4px;
  border: 1px solid #555;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 100;
}

.suggestion-item {
  padding: 8px;
  cursor: pointer;
  border-bottom: 1px solid #444;
  color: #ddd;
}

.suggestion-item:last-child {
  border-bottom: none;
}

.suggestion-item:hover {
  background-color: #444;
}

/* Results styling */
.results {
  margin-top: 10px;
  color: #cccccc;
  display: flex;
  flex-direction: column;
  height: calc(100% - 40px);
}

.message {
  color: #cccccc;
  margin-bottom: 10px;
}

.highlight-message {
  color: #0078d4;
}

.result-item {
  margin-bottom: 10px;
  padding: 5px;
  border-bottom: 1px solid #444;
}

.result-summary {
  background-color: #2a2a2a;
  padding: 10px 12px;
  border-radius: 4px;
  line-height: 1.5;
  border-left: 3px solid #0078d4;
  margin-bottom: 15px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: calc(100vh - 340px); /* Adjust for header, filter buttons, and bottom margin */
  overflow-y: auto;
  margin-top: 10px;
  padding-right: 5px;
  flex-grow: 1;
  min-height: 200px;
}

.result-button {
  display: block;
  width: 100%;
  text-align: left;
  padding: 10px;
  background-color: #2a2a2a;
  border: 1px solid #444;
  border-radius: 4px;
  color: #e0e0e0;
  cursor: pointer;
  transition: all 0.2s ease;
}

.result-button:hover {
  background-color: #3a3a3a;
  border-color: #555;
}

.result-button.selected {
  background-color: #0078d4;
  border-color: #0086f0;
}

/* Confidence indicator styling */
.confidence-indicator {
  display: inline-block;
  min-width: 45px;
  text-align: center;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: bold;
  margin-right: 8px;
  font-size: 0.85em;
}

.high-confidence {
  background-color: #10b981; /* Green */
  color: #000;
}

.medium-confidence {
  background-color: #f59e0b; /* Yellow/Amber */
  color: #000;
}

.low-confidence {
  background-color: #ef4444; /* Red */
  color: #fff;
}

.highlight-word {
  color: #0078d4;
  font-weight: bold;
}

.folder-name {
  color: #3b82f6;
  font-weight: bold;
}

.file-name {
  color: #10b981;
  font-weight: bold;
}

.page-number {
  color: #f59e0b;
  font-weight: bold;
}

.match-count {
  color: #ec4899;
  font-weight: bold;
  margin-left: 5px;
}

.detected-words {
  color: #a78bfa;
  font-style: italic;
  font-size: 0.9em;
  margin-left: 5px;
  display: block;
  margin-top: 4px;
}

/* Image container styling */
.image-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* PDF placeholder styling */
.pdf-placeholder {
  color: #777;
  font-style: italic;
  border: 2px dashed #555;
  padding: 100px;
  text-align: center;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Placeholder styling */
.placeholder {
  color: #777;
  font-style: italic;
  border: 2px dashed #555;
  padding: 100px;
  text-align: center;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>