import { useState } from "react";
import "./App.css";

type ImageItem = {
  image_url: string;
  title: string;
  text: string;
  id: string;
};

function App() {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

      const data: ImageItem[] = await response.json();
      setImages(data.slice(0, 20)); // Limit to 10 results
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async () => {
    if (!query) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/image-search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) throw new Error(`Error ${response.status}`);

      const data: ImageItem[] = await response.json();
      setImages(data.slice(0, 20)); // same as handleSearch
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Items Recommender</h1>
      <div className="search-bar">
        <input
          type="text"
          placeholder="Enter your query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Semantic Search"}
        </button>
        <button onClick={handleImageSearch} disabled={loading}>
          {loading ? "Searching..." : "Visual Search"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="image-grid">
        {images.map((item, idx) => (
          <div className="card" key={idx}>
            <p>{item.title}</p>
            <img src={item.image_url} alt={item.title} />
            <p>{item.id}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
