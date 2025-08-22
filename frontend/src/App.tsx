import { useState, useEffect } from "react";
import "./App.css";

type ImageItem = {
  png_base64: string;
  name: string;
};

type SearchResponse = {
  reasoning: string;
  results: ImageItem[];
};

function App() {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [reasoning, setReasoning] = useState("");
  const [displayedReasoning, setDisplayedReasoning] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageItem | null>(null);

  useEffect(() => {
    setDisplayedReasoning("");
    if (!reasoning) return;
    let i = 0;
    const interval = setInterval(() => {
      setDisplayedReasoning((prev) => prev + reasoning.charAt(i));
      i++;
      if (i >= reasoning.length) clearInterval(interval);
    }, 20);
    return () => clearInterval(interval);
  }, [reasoning]);

  const handleSearch = async () => {
    if (!query) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) throw new Error(`Error ${response.status}`);

      const data: SearchResponse = await response.json();
      setReasoning(data.reasoning);
      setImages(data.results.slice(0, 20)); // Limit to 20 results
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Who should I hire ?</h1>
      {reasoning && <pre className="reasoning">{displayedReasoning}</pre>}
      <div className="search-bar">
        <input
          type="text"
          placeholder="Enter your query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Find my next employee"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="image-grid">
        {images.map((item, idx) => (
          <div className="card" key={idx} onClick={() => setSelectedImage(item)}>
            <p>{item.name}</p>
            <img
              src={`data:image/png;base64,${item.png_base64}`}
              alt={item.name}
            />
          </div>
        ))}
      </div>

      {selectedImage && (
        <div className="modal" onClick={() => setSelectedImage(null)}>
          <button
            className="close-btn"
            onClick={(e) => {
              e.stopPropagation();
              setSelectedImage(null);
            }}
          >
            ×
          </button>
          <img
            src={`data:image/png;base64,${selectedImage.png_base64}`}
            alt={selectedImage.name}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

export default App;
