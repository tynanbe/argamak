defmodule Argamak.MixProject do
  use Mix.Project

  @app :argamak
  @version "0.1.0"

  def project do
    [
      app: @app,
      version: @version,
      elixir: "~> 1.12",
      name: "#{@app}",
      description: "A tensor library for the Gleam programming language",
      package: package(),
      archives: [mix_gleam: "~> 0.4.0"],
      aliases: MixGleam.add_aliases(),
      erlc_paths: ["build/dev/erlang/#{@app}/build"],
      erlc_include_path: "build/dev/erlang/#{@app}/include",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:gleam_stdlib, "~> 0.19"},
      {:gleam_erlang, "~> 0.8"},
      # {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx"},
      {:gleeunit, "~> 0.6", only: [:dev, :test], runtime: false},
    ]
  end

  defp package do
    [
      files: [
        "gleam.toml",
        "LICENSE",
        "mix.exs",
        "NOTICE",
        "README.md",
        "src",
      ],
      licenses: ["Apache-2.0"],
      links: %{
        "Repository" => "https://github.com/tynanbe/argamak",
        "Website" => "https://gleam.run/",
      },
      maintainers: ["Tynan Beatty"],
    ]
  end
end
