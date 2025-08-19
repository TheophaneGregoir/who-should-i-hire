import { useState } from "react";
import "./App.css";

type ImageItem = {
  path_to_png: string;
  name: string;
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
      setImages(data.slice(0, 20)); // Limit to 20 results
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
      </div>

      {error && <p className="error">{error}</p>}

      <div className="image-grid">
        {images.map((item, idx) => (
          <div className="card" key={idx}>
            <p>{item.name}</p>
            <img src={item.path_to_png} alt={item.name} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
