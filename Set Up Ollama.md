# Ollama

Ollama is a tool for running large language models (LLMs) locally on your machine. Setting up Ollama varies slightly depending on your operating system. Below is a summary for the main platforms, with official sources for each.

---

## 1. macOS

**Installation:**

- You can install Ollama on macOS using Homebrew or by downloading the official installer.

**Using Homebrew:**

```sh
brew install ollama
```

**Or download the installer:**

- Visit: [https://ollama.com/download](https://ollama.com/download)

**Source:**  

- [Ollama macOS Install Guide](https://github.com/ollama/ollama/blob/main/docs/install.md#macos)

---

## 2. Windows

**Installation:**

- Download the official installer from the Ollama website.

**Steps:**

1. Go to [https://ollama.com/download](https://ollama.com/download)
2. Download the Windows installer (.exe)
3. Run the installer and follow the prompts.

**Source:**  

- [Ollama Windows Install Guide](https://github.com/ollama/ollama/blob/main/docs/install.md#windows)

---

## 3. Linux

**Installation:**

- Use the provided script or download the binary.

**Using the install script:**

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

**Or download the binary:**

- Visit: [https://ollama.com/download](https://ollama.com/download) and select the Linux version.

**Source:**  

- [Ollama Linux Install Guide](https://github.com/ollama/ollama/blob/main/docs/install.md#linux)

---

## 4. Docker (for any platform)

Ollama can also be run in a Docker container, which is useful for platforms not directly supported or for isolated environments.

**Example:**

```sh
docker run -d -p 11434:11434 ollama/ollama
```

**Source:**  

- [Ollama Docker Guide](https://github.com/ollama/ollama/blob/main/docs/docker.md)

---

## Additional Resources

- **Official Documentation:** [https://github.com/ollama/ollama/blob/main/docs/install.md](https://github.com/ollama/ollama/blob/main/docs/install.md)
- **Ollama Website:** [https://ollama.com/](https://ollama.com/)

---

If you need platform-specific troubleshooting or advanced setup (e.g., GPU support, running as a service), refer to the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) for more details.
